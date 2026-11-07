-- One-time setup uses fast bulk-import settings.
PRAGMA journal_mode = OFF;
PRAGMA synchronous = OFF;
PRAGMA cache_size = -100000;
PRAGMA mmap_size = 1073741824;
-- SmolLM-135M, SQL-only SQLite foundation.
-- Run with the sqlite3 command-line program, which supplies readfile().
-- Model source: https://huggingface.co/HuggingFaceTB/SmolLM-135M
-- model.safetensors SHA-256:
-- c7a387d6fe81ca6dd304aeb809bda3932ff1bbef3ca41c9484502f2f448dc093
-- The model.safetensors file remains data; no code outside SQL parses it.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS model_file (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  bytes BLOB NOT NULL CHECK (typeof(bytes) = 'blob')
);

CREATE TABLE IF NOT EXISTS tensor (
  name TEXT PRIMARY KEY,
  dtype TEXT NOT NULL CHECK (dtype IN ('F32', 'BF16', 'F16')),
  shape TEXT NOT NULL CHECK (json_valid(shape)),
  elements INTEGER NOT NULL CHECK (elements > 0),
  bytes BLOB NOT NULL CHECK (typeof(bytes) = 'blob'),
  CHECK (length(bytes) = elements *
    CASE dtype WHEN 'F32' THEN 4 ELSE 2 END)
);

-- A caller writes the requested scalar coordinates here. The scalar_value
-- view interprets the model's IEEE-754 data without a host-language UDF.
CREATE TABLE IF NOT EXISTS scalar_request (
  name TEXT NOT NULL REFERENCES tensor(name),
  linear_index INTEGER NOT NULL CHECK (linear_index >= 0),
  PRIMARY KEY (name, linear_index)
) WITHOUT ROWID;

-- Safetensors stores an unsigned 64-bit little-endian JSON-header length.
-- A valid model header is much smaller than 4 GiB; reject larger headers.
CREATE VIEW IF NOT EXISTS model_header AS
WITH first_bytes AS (
  SELECT bytes, hex(substr(bytes, 1, 8)) AS h FROM model_file WHERE id = 1
), nibbles AS (
  SELECT bytes, h,
    instr('0123456789ABCDEF', substr(h, 1, 1)) - 1 AS n1,
    instr('0123456789ABCDEF', substr(h, 2, 1)) - 1 AS n2,
    instr('0123456789ABCDEF', substr(h, 3, 1)) - 1 AS n3,
    instr('0123456789ABCDEF', substr(h, 4, 1)) - 1 AS n4,
    instr('0123456789ABCDEF', substr(h, 5, 1)) - 1 AS n5,
    instr('0123456789ABCDEF', substr(h, 6, 1)) - 1 AS n6,
    instr('0123456789ABCDEF', substr(h, 7, 1)) - 1 AS n7,
    instr('0123456789ABCDEF', substr(h, 8, 1)) - 1 AS n8
  FROM first_bytes
), lengths AS (
  SELECT bytes,
    n1 * 16 + n2 + 256 * (n3 * 16 + n4) +
    65536 * (n5 * 16 + n6) + 16777216 * (n7 * 16 + n8)
      AS header_length,
    substr(h, 9) AS high_bytes
  FROM nibbles
)
SELECT header_length, CAST(substr(bytes, 9, header_length) AS TEXT) AS json,
       length(bytes) AS file_length
FROM lengths
WHERE high_bytes = '00000000'
  AND header_length > 0
  AND header_length + 8 <= length(bytes);

CREATE VIEW IF NOT EXISTS safetensors_manifest AS
SELECT j.key AS name,
       json_extract(j.value, '$.dtype') AS dtype,
       json_extract(j.value, '$.shape') AS shape,
       json_extract(j.value, '$.data_offsets[0]') AS start_offset,
       json_extract(j.value, '$.data_offsets[1]') AS end_offset,
       h.header_length
FROM model_header AS h, json_each(h.json) AS j
WHERE j.key <> '__metadata__';

CREATE VIEW IF NOT EXISTS scalar_bytes AS
WITH raw AS (
  SELECT r.name, r.linear_index, t.dtype,
         hex(substr(t.bytes,
           1 + r.linear_index * CASE t.dtype WHEN 'F32' THEN 4 ELSE 2 END,
           CASE t.dtype WHEN 'F32' THEN 4 ELSE 2 END)) AS h
  FROM scalar_request AS r
  JOIN tensor AS t ON t.name = r.name
  WHERE r.linear_index < t.elements
)
SELECT name, linear_index, dtype,
       16 * (instr('0123456789ABCDEF', substr(h, 1, 1)) - 1) +
            (instr('0123456789ABCDEF', substr(h, 2, 1)) - 1) AS b0,
       16 * (instr('0123456789ABCDEF', substr(h, 3, 1)) - 1) +
            (instr('0123456789ABCDEF', substr(h, 4, 1)) - 1) AS b1,
       16 * (instr('0123456789ABCDEF', substr(h, 5, 1)) - 1) +
            (instr('0123456789ABCDEF', substr(h, 6, 1)) - 1) AS b2,
       16 * (instr('0123456789ABCDEF', substr(h, 7, 1)) - 1) +
            (instr('0123456789ABCDEF', substr(h, 8, 1)) - 1) AS b3
FROM raw;

CREATE VIEW IF NOT EXISTS scalar_bits AS
SELECT name, linear_index, dtype,
       CASE dtype WHEN 'F32'
         THEN b0 + 256 * b1 + 65536 * b2 + 16777216 * b3
         ELSE b0 + 256 * b1 END AS bits
FROM scalar_bytes;

CREATE VIEW IF NOT EXISTS scalar_value AS
WITH parts AS (
  SELECT name, linear_index, dtype, bits,
         CASE dtype WHEN 'F32' THEN (bits >> 31) & 1
                    ELSE (bits >> 15) & 1 END AS sign_bit,
         CASE dtype WHEN 'F32' THEN (bits >> 23) & 255
                    WHEN 'BF16' THEN (bits >> 7) & 255
                    ELSE (bits >> 10) & 31 END AS exponent,
         CASE dtype WHEN 'F32' THEN bits & 8388607
                    WHEN 'BF16' THEN bits & 127
                    ELSE bits & 1023 END AS fraction
  FROM scalar_bits
)
SELECT name, linear_index,
       CASE
         WHEN dtype = 'F16' AND exponent = 31 THEN NULL
         WHEN dtype <> 'F16' AND exponent = 255 THEN NULL
         WHEN exponent = 0 THEN
           (1 - 2 * sign_bit) * fraction *
           pow(2.0, CASE dtype WHEN 'F32' THEN -149
                               WHEN 'BF16' THEN -133 ELSE -24 END)
         ELSE (1 - 2 * sign_bit) *
           (1.0 + fraction /
             CASE dtype WHEN 'F32' THEN 8388608.0
                        WHEN 'BF16' THEN 128.0 ELSE 1024.0 END) *
           pow(2.0, exponent -
             CASE dtype WHEN 'F16' THEN 15 ELSE 127 END)
       END AS value
FROM parts;
INSERT INTO model_file(id, bytes)
VALUES (1, readfile('outputs/model.safetensors'));
-- After loading model_file, import its tensors into the SQL model store.
-- Example: INSERT INTO model_file VALUES (1, readfile('outputs/model.safetensors'));
BEGIN;
INSERT OR REPLACE INTO tensor(name, dtype, shape, elements, bytes)
SELECT m.name, m.dtype, m.shape,
       (m.end_offset - m.start_offset) /
         CASE m.dtype WHEN 'F32' THEN 4 ELSE 2 END,
       substr(f.bytes, 9 + m.header_length + m.start_offset,
              m.end_offset - m.start_offset)
FROM safetensors_manifest AS m
CROSS JOIN model_file AS f
WHERE f.id = 1
  AND m.dtype IN ('F32', 'BF16', 'F16')
  AND m.end_offset > m.start_offset
  AND m.end_offset + m.header_length + 8 <= length(f.bytes)
  AND (m.end_offset - m.start_offset) %
        CASE m.dtype WHEN 'F32' THEN 4 ELSE 2 END = 0;
CREATE TEMP TABLE IF NOT EXISTS import_validation (
  ok INTEGER NOT NULL CHECK (ok = 1)
);
INSERT OR ROLLBACK INTO import_validation(ok)
SELECT CASE WHEN
  (SELECT count(*) FROM tensor) = (SELECT count(*) FROM safetensors_manifest)
  AND (SELECT count(*) FROM tensor) > 0
THEN 1 ELSE 0 END;
DELETE FROM model_file;
COMMIT;
-- Store each tensor row as a small BLOB so scalar reads never touch a whole
-- multi-megabyte tensor BLOB. Run once after 02_import.sql.
CREATE TABLE IF NOT EXISTS tensor_row (
  name TEXT NOT NULL REFERENCES tensor(name),
  row_index INTEGER NOT NULL CHECK (row_index >= 0),
  bytes BLOB NOT NULL CHECK (typeof(bytes) = 'blob'),
  PRIMARY KEY (name, row_index)
) WITHOUT ROWID;

INSERT OR REPLACE INTO tensor_row(name, row_index, bytes)
WITH RECURSIVE rows(row_index) AS (
  SELECT 0
  UNION ALL
  SELECT row_index + 1 FROM rows WHERE row_index < 49151
)
SELECT t.name, r.row_index,
  substr(t.bytes,
    1 + r.row_index *
      CASE WHEN json_array_length(t.shape) = 2
        THEN json_extract(t.shape, '$[1]') ELSE t.elements END *
      CASE t.dtype WHEN 'F32' THEN 4 ELSE 2 END,
    CASE WHEN json_array_length(t.shape) = 2
      THEN json_extract(t.shape, '$[1]') ELSE t.elements END *
    CASE t.dtype WHEN 'F32' THEN 4 ELSE 2 END)
FROM tensor AS t CROSS JOIN rows AS r
WHERE r.row_index <
  CASE WHEN json_array_length(t.shape) = 2
    THEN json_extract(t.shape, '$[0]') ELSE 1 END;

DROP VIEW scalar_bytes;
CREATE VIEW scalar_bytes AS
WITH raw AS (
  SELECT r.name, r.linear_index, t.dtype,
         hex(substr(tr.bytes,
           1 + (r.linear_index %
             CASE WHEN json_array_length(t.shape) = 2
               THEN json_extract(t.shape, '$[1]') ELSE t.elements END) *
             CASE t.dtype WHEN 'F32' THEN 4 ELSE 2 END,
           CASE t.dtype WHEN 'F32' THEN 4 ELSE 2 END)) AS h
  FROM scalar_request AS r
  JOIN tensor AS t ON t.name = r.name
  JOIN tensor_row AS tr
    ON tr.name = r.name
   AND tr.row_index = r.linear_index /
     CASE WHEN json_array_length(t.shape) = 2
       THEN json_extract(t.shape, '$[1]') ELSE t.elements END
  WHERE r.linear_index < t.elements
)
SELECT name, linear_index, dtype,
       16 * (instr('0123456789ABCDEF', substr(h, 1, 1)) - 1) +
            (instr('0123456789ABCDEF', substr(h, 2, 1)) - 1) AS b0,
       16 * (instr('0123456789ABCDEF', substr(h, 3, 1)) - 1) +
            (instr('0123456789ABCDEF', substr(h, 4, 1)) - 1) AS b1,
       16 * (instr('0123456789ABCDEF', substr(h, 5, 1)) - 1) +
            (instr('0123456789ABCDEF', substr(h, 6, 1)) - 1) AS b2,
       16 * (instr('0123456789ABCDEF', substr(h, 7, 1)) - 1) +
            (instr('0123456789ABCDEF', substr(h, 8, 1)) - 1) AS b3
FROM raw;
-- Import the published tokenizer.json entirely with SQL.
-- Example: INSERT INTO tokenizer_file VALUES (1, readfile('work/tokenizer.json'));
CREATE TABLE IF NOT EXISTS tokenizer_file (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  bytes BLOB NOT NULL CHECK (typeof(bytes) = 'blob')
);

CREATE TABLE IF NOT EXISTS tokenizer_vocab (
  symbol TEXT PRIMARY KEY,
  token_id INTEGER NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS tokenizer_merge (
  rank INTEGER PRIMARY KEY,
  left_symbol TEXT NOT NULL,
  right_symbol TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS tokenizer_merge_pair
ON tokenizer_merge(left_symbol, right_symbol);

CREATE TABLE IF NOT EXISTS tokenizer_special (
  token_id INTEGER PRIMARY KEY,
  content TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS byte_symbol (
  byte_value INTEGER PRIMARY KEY CHECK (byte_value BETWEEN 0 AND 255),
  symbol TEXT NOT NULL UNIQUE
);

WITH RECURSIVE byte_range(b) AS (
  SELECT 0 UNION ALL SELECT b + 1 FROM byte_range WHERE b < 255
), tagged AS (
  SELECT b,
    CASE WHEN b BETWEEN 33 AND 126 OR b BETWEEN 161 AND 172
              OR b BETWEEN 174 AND 255 THEN 1 ELSE 0 END AS direct
  FROM byte_range
), numbered AS (
  SELECT b, direct,
    sum(CASE WHEN direct = 0 THEN 1 ELSE 0 END)
      OVER (ORDER BY b) - 1 AS extra_index
  FROM tagged
)
INSERT OR REPLACE INTO byte_symbol(byte_value, symbol)
SELECT b, char(CASE WHEN direct = 1 THEN b ELSE 256 + extra_index END)
FROM numbered;

INSERT INTO tokenizer_file(id, bytes)
VALUES (1, readfile('outputs/tokenizer.json'));
-- Run after inserting tokenizer_file(1, readfile('outputs/tokenizer.json')).
BEGIN;
INSERT OR REPLACE INTO tokenizer_vocab(symbol, token_id)
SELECT j.key, CAST(j.value AS INTEGER)
FROM tokenizer_file AS f, json_each(CAST(f.bytes AS TEXT), '$.model.vocab') AS j
WHERE f.id = 1;

INSERT OR REPLACE INTO tokenizer_merge(rank, left_symbol, right_symbol)
SELECT CAST(j.key AS INTEGER),
       substr(j.value, 1, instr(j.value, ' ') - 1),
       substr(j.value, instr(j.value, ' ') + 1)
FROM tokenizer_file AS f, json_each(CAST(f.bytes AS TEXT), '$.model.merges') AS j
WHERE f.id = 1 AND instr(j.value, ' ') > 0;
INSERT OR REPLACE INTO tokenizer_special(token_id, content)
SELECT json_extract(j.value, '$.id'), json_extract(j.value, '$.content')
FROM tokenizer_file AS f, json_each(CAST(f.bytes AS TEXT), '$.added_tokens') AS j
WHERE f.id = 1 AND json_extract(j.value, '$.special') = 1;
DELETE FROM tokenizer_file;
COMMIT;
-- SQL-only matrix-vector multiplication over a safetensors matrix.
-- Set matrix_input, then INSERT INTO matrix_job VALUES (1, tensor_name).
-- The INSERT trigger fills matrix_output. Rows are zero-based.
CREATE TABLE IF NOT EXISTS matrix_input (
  dim INTEGER PRIMARY KEY CHECK (dim >= 0),
  value REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS matrix_job (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  tensor_name TEXT NOT NULL REFERENCES tensor(name),
  first_row INTEGER NOT NULL DEFAULT 0 CHECK (first_row >= 0),
  end_row INTEGER CHECK (end_row > first_row)
);

CREATE TABLE IF NOT EXISTS matrix_output (
  dim INTEGER PRIMARY KEY CHECK (dim >= 0),
  value REAL NOT NULL
);

CREATE TRIGGER IF NOT EXISTS run_matrix_job
AFTER INSERT ON matrix_job
BEGIN
  SELECT RAISE(ABORT, 'matrix input dimension mismatch')
  WHERE (SELECT count(*) FROM matrix_input) <>
    (SELECT json_extract(shape, '$[1]') FROM tensor
     WHERE name = NEW.tensor_name)
    OR (SELECT min(dim) FROM matrix_input) <> 0
    OR (SELECT max(dim) FROM matrix_input) <>
      (SELECT json_extract(shape, '$[1]') - 1 FROM tensor
       WHERE name = NEW.tensor_name);
  DELETE FROM scalar_request;
  INSERT INTO scalar_request(name, linear_index)
  WITH RECURSIVE rows(r) AS (
    SELECT NEW.first_row
    UNION ALL
    SELECT r + 1 FROM rows
    WHERE r + 1 < coalesce(NEW.end_row,
      (SELECT json_extract(t.shape, '$[0]')
       FROM tensor AS t WHERE t.name = NEW.tensor_name))
  )
  SELECT NEW.tensor_name,
         r * json_extract(t.shape, '$[1]') + i.dim
  FROM rows
  JOIN tensor AS t ON t.name = NEW.tensor_name
  JOIN matrix_input AS i
  WHERE json_array_length(t.shape) = 2
    AND i.dim < json_extract(t.shape, '$[1]');

  DELETE FROM matrix_output;
  INSERT INTO matrix_output(dim, value)
  SELECT CAST(s.linear_index / json_extract(t.shape, '$[1]') AS INTEGER),
         sum(s.value * i.value)
  FROM scalar_value AS s
  JOIN tensor AS t ON t.name = s.name
  JOIN matrix_input AS i
    ON i.dim = s.linear_index % json_extract(t.shape, '$[1]')
  WHERE s.name = NEW.tensor_name
  GROUP BY 1;
  SELECT RAISE(ABORT, 'matrix output row mismatch')
  WHERE (SELECT count(*) FROM matrix_output) <>
    (coalesce(NEW.end_row,
      (SELECT json_extract(shape, '$[0]') FROM tensor
       WHERE name = NEW.tensor_name)) - NEW.first_row);
END;
-- Expected HuggingFaceTB/SmolLM-135M tensor layout (config.json).
CREATE VIEW IF NOT EXISTS expected_tensor AS
WITH RECURSIVE layer(n) AS (
  SELECT 0 UNION ALL SELECT n + 1 FROM layer WHERE n < 29
), suffix(s, shape) AS (
  VALUES
    ('input_layernorm.weight', '[576]'),
    ('post_attention_layernorm.weight', '[576]'),
    ('self_attn.q_proj.weight', '[576,576]'),
    ('self_attn.k_proj.weight', '[192,576]'),
    ('self_attn.v_proj.weight', '[192,576]'),
    ('self_attn.o_proj.weight', '[576,576]'),
    ('mlp.gate_proj.weight', '[1536,576]'),
    ('mlp.up_proj.weight', '[1536,576]'),
    ('mlp.down_proj.weight', '[576,1536]')
)
SELECT 'model.embed_tokens.weight' AS name, '[49152,576]' AS shape
UNION ALL SELECT 'model.norm.weight', '[576]'
UNION ALL SELECT 'model.layers.' || layer.n || '.' || suffix.s, suffix.shape
FROM layer CROSS JOIN suffix;

CREATE VIEW IF NOT EXISTS model_problem AS
SELECT e.name, 'missing or wrong shape/dtype' AS problem
FROM expected_tensor AS e
LEFT JOIN tensor AS t ON t.name = e.name
WHERE t.name IS NULL OR t.shape <> e.shape OR t.dtype <> 'F32'
UNION ALL
SELECT t.name, 'unexpected tensor' FROM tensor AS t
LEFT JOIN expected_tensor AS e ON e.name = t.name
WHERE e.name IS NULL;
-- One-token SmolLM-135M forward pass. Insert token IDs at consecutive positions
-- into inference_input. All arithmetic and control flow run in SQLite SQL.
PRAGMA recursive_triggers = ON;

CREATE TABLE IF NOT EXISTS inference_input (
  pos INTEGER PRIMARY KEY CHECK (pos BETWEEN 0 AND 2047),
  token_id INTEGER NOT NULL CHECK (token_id BETWEEN 0 AND 49151)
);
CREATE TABLE IF NOT EXISTS hidden (dim INTEGER PRIMARY KEY, value REAL NOT NULL);
CREATE TABLE IF NOT EXISTS norm_vector (dim INTEGER PRIMARY KEY, value REAL NOT NULL);
CREATE TABLE IF NOT EXISTS q_raw (dim INTEGER PRIMARY KEY, value REAL NOT NULL);
CREATE TABLE IF NOT EXISTS k_raw (dim INTEGER PRIMARY KEY, value REAL NOT NULL);
CREATE TABLE IF NOT EXISTS v_raw (dim INTEGER PRIMARY KEY, value REAL NOT NULL);
CREATE TABLE IF NOT EXISTS q_rot (dim INTEGER PRIMARY KEY, value REAL NOT NULL);
CREATE TABLE IF NOT EXISTS k_rot (dim INTEGER PRIMARY KEY, value REAL NOT NULL);
CREATE TABLE IF NOT EXISTS key_cache (
  layer INTEGER NOT NULL, pos INTEGER NOT NULL,
  kv_head INTEGER NOT NULL, dim INTEGER NOT NULL,
  value REAL NOT NULL,
  PRIMARY KEY (layer, pos, kv_head, dim)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS value_cache (
  layer INTEGER NOT NULL, pos INTEGER NOT NULL,
  kv_head INTEGER NOT NULL, dim INTEGER NOT NULL,
  value REAL NOT NULL,
  PRIMARY KEY (layer, pos, kv_head, dim)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS attention_score (
  head INTEGER NOT NULL, context_pos INTEGER NOT NULL,
  score REAL NOT NULL,
  PRIMARY KEY (head, context_pos)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS attention_prob (
  head INTEGER NOT NULL, context_pos INTEGER NOT NULL,
  prob REAL NOT NULL,
  PRIMARY KEY (head, context_pos)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS attention_context (
  dim INTEGER PRIMARY KEY, value REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS gate_vector (dim INTEGER PRIMARY KEY, value REAL NOT NULL);
CREATE TABLE IF NOT EXISTS up_vector (dim INTEGER PRIMARY KEY, value REAL NOT NULL);
CREATE TABLE IF NOT EXISTS layer_job (
  pos INTEGER NOT NULL, layer INTEGER NOT NULL CHECK (layer BETWEEN 0 AND 29),
  PRIMARY KEY (pos, layer)
);
CREATE TABLE IF NOT EXISTS layer_done (
  pos INTEGER NOT NULL, layer INTEGER NOT NULL,
  PRIMARY KEY (pos, layer)
);
CREATE TABLE IF NOT EXISTS inference_settings (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  last_layer INTEGER NOT NULL CHECK (last_layer BETWEEN 0 AND 29),
  emit_logits INTEGER NOT NULL DEFAULT 1 CHECK (emit_logits IN (0,1))
);
INSERT OR IGNORE INTO inference_settings VALUES (1, 29, 1);
CREATE TABLE IF NOT EXISTS final_job (pos INTEGER PRIMARY KEY);
CREATE TABLE IF NOT EXISTS logit_chunk (
  pos INTEGER NOT NULL, first_row INTEGER NOT NULL,
  PRIMARY KEY (pos, first_row)
);
CREATE TABLE IF NOT EXISTS logit_score (
  token_id INTEGER PRIMARY KEY, value REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS generated (
  pos INTEGER PRIMARY KEY,
  next_token_id INTEGER NOT NULL,
  logit REAL NOT NULL
);

CREATE TRIGGER IF NOT EXISTS start_inference
AFTER INSERT ON inference_input
BEGIN
  DELETE FROM scalar_request;
  INSERT INTO scalar_request(name, linear_index)
  WITH RECURSIVE dims(d) AS (
    SELECT 0 UNION ALL SELECT d + 1 FROM dims WHERE d < 575
  )
  SELECT 'model.embed_tokens.weight', NEW.token_id * 576 + d FROM dims;
  DELETE FROM hidden;
  INSERT INTO hidden(dim, value)
  SELECT linear_index % 576, value FROM scalar_value
  WHERE name = 'model.embed_tokens.weight';
  INSERT INTO layer_job(pos, layer) VALUES (NEW.pos, 0);
  SELECT RAISE(ABORT, 'recursive layer triggers did not execute')
  WHERE (SELECT count(*) FROM layer_done WHERE pos = NEW.pos) <>
    (SELECT last_layer + 1 FROM inference_settings WHERE id = 1);
END;

CREATE TRIGGER IF NOT EXISTS run_layer
AFTER INSERT ON layer_job
BEGIN
  -- Attention input RMSNorm.
  DELETE FROM scalar_request;
  INSERT INTO scalar_request(name, linear_index)
  SELECT 'model.layers.' || NEW.layer || '.input_layernorm.weight', dim
  FROM hidden;
  DELETE FROM norm_vector;
  INSERT INTO norm_vector(dim, value)
  SELECT s.linear_index, h.value * s.value /
    sqrt((SELECT avg(value * value) FROM hidden) + 0.00001)
  FROM scalar_value AS s JOIN hidden AS h ON h.dim = s.linear_index;

  -- Query, key, and value projections.
  DELETE FROM matrix_input;
  INSERT INTO matrix_input SELECT * FROM norm_vector;
  DELETE FROM matrix_job;
  INSERT INTO matrix_job(id, tensor_name) VALUES
    (1, 'model.layers.' || NEW.layer || '.self_attn.q_proj.weight');
  DELETE FROM q_raw;
  INSERT INTO q_raw SELECT * FROM matrix_output;
  DELETE FROM matrix_job;
  INSERT INTO matrix_job(id, tensor_name) VALUES
    (1, 'model.layers.' || NEW.layer || '.self_attn.k_proj.weight');
  DELETE FROM k_raw;
  INSERT INTO k_raw SELECT * FROM matrix_output;
  DELETE FROM matrix_job;
  INSERT INTO matrix_job(id, tensor_name) VALUES
    (1, 'model.layers.' || NEW.layer || '.self_attn.v_proj.weight');
  DELETE FROM v_raw;
  INSERT INTO v_raw SELECT * FROM matrix_output;

  -- Llama RoPE rotates the first and second 32-dimensional half of each head.
  DELETE FROM q_rot;
  INSERT INTO q_rot(dim, value)
  SELECT q.dim,
    q.value * cos(NEW.pos * pow(10000.0, -(
      CASE WHEN q.dim % 64 < 32 THEN q.dim % 64
           ELSE q.dim % 64 - 32 END) / 32.0)) +
    (CASE WHEN q.dim % 64 < 32 THEN -pair.value ELSE pair.value END) *
      sin(NEW.pos * pow(10000.0, -(
      CASE WHEN q.dim % 64 < 32 THEN q.dim % 64
           ELSE q.dim % 64 - 32 END) / 32.0))
  FROM q_raw AS q JOIN q_raw AS pair
    ON pair.dim = CASE WHEN q.dim % 64 < 32 THEN q.dim + 32
                       ELSE q.dim - 32 END;
  DELETE FROM k_rot;
  INSERT INTO k_rot(dim, value)
  SELECT k.dim,
    k.value * cos(NEW.pos * pow(10000.0, -(
      CASE WHEN k.dim % 64 < 32 THEN k.dim % 64
           ELSE k.dim % 64 - 32 END) / 32.0)) +
    (CASE WHEN k.dim % 64 < 32 THEN -pair.value ELSE pair.value END) *
      sin(NEW.pos * pow(10000.0, -(
      CASE WHEN k.dim % 64 < 32 THEN k.dim % 64
           ELSE k.dim % 64 - 32 END) / 32.0))
  FROM k_raw AS k JOIN k_raw AS pair
    ON pair.dim = CASE WHEN k.dim % 64 < 32 THEN k.dim + 32
                       ELSE k.dim - 32 END;
  INSERT INTO key_cache(layer, pos, kv_head, dim, value)
  SELECT NEW.layer, NEW.pos, dim / 64, dim % 64, value FROM k_rot;
  INSERT INTO value_cache(layer, pos, kv_head, dim, value)
  SELECT NEW.layer, NEW.pos, dim / 64, dim % 64, value FROM v_raw;

  -- Grouped-query causal attention (three query heads per KV head).
  DELETE FROM attention_score;
  INSERT INTO attention_score(head, context_pos, score)
  SELECT q.dim / 64, k.pos, sum(q.value * k.value) / 8.0
  FROM q_rot AS q JOIN key_cache AS k
    ON k.layer = NEW.layer AND k.pos <= NEW.pos
   AND k.kv_head = (q.dim / 64) / 3 AND k.dim = q.dim % 64
  GROUP BY q.dim / 64, k.pos;
  DELETE FROM attention_prob;
  INSERT INTO attention_prob(head, context_pos, prob)
  WITH shifted AS (
    SELECT head, context_pos,
      exp(score - max(score) OVER (PARTITION BY head)) AS e
    FROM attention_score
  )
  SELECT head, context_pos, e / sum(e) OVER (PARTITION BY head)
  FROM shifted;
  DELETE FROM attention_context;
  INSERT INTO attention_context(dim, value)
  SELECT p.head * 64 + v.dim, sum(p.prob * v.value)
  FROM attention_prob AS p JOIN value_cache AS v
    ON v.layer = NEW.layer AND v.pos = p.context_pos
   AND v.kv_head = p.head / 3
  GROUP BY p.head * 64 + v.dim;

  -- Output projection and first residual addition.
  DELETE FROM matrix_input;
  INSERT INTO matrix_input SELECT * FROM attention_context;
  DELETE FROM matrix_job;
  INSERT INTO matrix_job(id, tensor_name) VALUES
    (1, 'model.layers.' || NEW.layer || '.self_attn.o_proj.weight');
  UPDATE hidden SET value = value +
    (SELECT o.value FROM matrix_output AS o WHERE o.dim = hidden.dim);

  -- Feed-forward input RMSNorm.
  DELETE FROM scalar_request;
  INSERT INTO scalar_request(name, linear_index)
  SELECT 'model.layers.' || NEW.layer || '.post_attention_layernorm.weight',
         dim FROM hidden;
  DELETE FROM norm_vector;
  INSERT INTO norm_vector(dim, value)
  SELECT s.linear_index, h.value * s.value /
    sqrt((SELECT avg(value * value) FROM hidden) + 0.00001)
  FROM scalar_value AS s JOIN hidden AS h ON h.dim = s.linear_index;

  -- SwiGLU feed-forward network and second residual addition.
  DELETE FROM matrix_input;
  INSERT INTO matrix_input SELECT * FROM norm_vector;
  DELETE FROM matrix_job;
  INSERT INTO matrix_job(id, tensor_name) VALUES
    (1, 'model.layers.' || NEW.layer || '.mlp.gate_proj.weight');
  DELETE FROM gate_vector;
  INSERT INTO gate_vector SELECT * FROM matrix_output;
  DELETE FROM matrix_job;
  INSERT INTO matrix_job(id, tensor_name) VALUES
    (1, 'model.layers.' || NEW.layer || '.mlp.up_proj.weight');
  DELETE FROM up_vector;
  INSERT INTO up_vector SELECT * FROM matrix_output;
  DELETE FROM matrix_input;
  INSERT INTO matrix_input(dim, value)
  SELECT g.dim, g.value * u.value / (1.0 + exp(-g.value))
  FROM gate_vector AS g JOIN up_vector AS u ON u.dim = g.dim;
  DELETE FROM matrix_job;
  INSERT INTO matrix_job(id, tensor_name) VALUES
    (1, 'model.layers.' || NEW.layer || '.mlp.down_proj.weight');
  UPDATE hidden SET value = value +
    (SELECT o.value FROM matrix_output AS o WHERE o.dim = hidden.dim);

  INSERT INTO layer_done(pos, layer) VALUES (NEW.pos, NEW.layer);

  INSERT INTO layer_job(pos, layer)
  SELECT NEW.pos, NEW.layer + 1
  WHERE NEW.layer < (SELECT last_layer FROM inference_settings WHERE id = 1);
  INSERT INTO final_job(pos)
  SELECT NEW.pos
  WHERE NEW.layer = (SELECT last_layer FROM inference_settings WHERE id = 1)
    AND (SELECT emit_logits FROM inference_settings WHERE id = 1) = 1;
END;

CREATE TRIGGER IF NOT EXISTS run_final_norm
AFTER INSERT ON final_job
BEGIN
  DELETE FROM scalar_request;
  INSERT INTO scalar_request(name, linear_index)
  SELECT 'model.norm.weight', dim FROM hidden;
  DELETE FROM matrix_input;
  INSERT INTO matrix_input(dim, value)
  SELECT s.linear_index, h.value * s.value /
    sqrt((SELECT avg(value * value) FROM hidden) + 0.00001)
  FROM scalar_value AS s JOIN hidden AS h ON h.dim = s.linear_index;
  DELETE FROM logit_score;
  INSERT INTO logit_chunk(pos, first_row) VALUES (NEW.pos, 0);
END;

CREATE TRIGGER IF NOT EXISTS run_logit_chunk
AFTER INSERT ON logit_chunk
BEGIN
  DELETE FROM matrix_job;
  INSERT INTO matrix_job(id, tensor_name, first_row, end_row) VALUES
    (1, 'model.embed_tokens.weight', NEW.first_row,
     min(49152, NEW.first_row + 256));
  INSERT INTO logit_score(token_id, value)
  SELECT dim, value FROM matrix_output;
  INSERT INTO logit_chunk(pos, first_row)
  SELECT NEW.pos, NEW.first_row + 256
  WHERE NEW.first_row + 256 < 49152;
  INSERT INTO generated(pos, next_token_id, logit)
  SELECT NEW.pos, token_id, value FROM logit_score
  WHERE NEW.first_row + 256 >= 49152
  ORDER BY value DESC, token_id ASC LIMIT 1;
END;
-- Byte-level BPE on one already pretokenized piece. SQLite SQL only.
-- Insert (piece_id, text) into bpe_piece, then read bpe_tokens.
PRAGMA recursive_triggers = ON;
CREATE TABLE IF NOT EXISTS bpe_piece (
  piece_id INTEGER PRIMARY KEY,
  text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bpe_symbol (
  piece_id INTEGER NOT NULL,
  position INTEGER NOT NULL,
  symbol TEXT NOT NULL,
  PRIMARY KEY (piece_id, position)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS bpe_step (
  piece_id INTEGER NOT NULL,
  iteration INTEGER NOT NULL,
  PRIMARY KEY (piece_id, iteration)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS bpe_choice (
  piece_id INTEGER PRIMARY KEY,
  left_position INTEGER NOT NULL,
  right_position INTEGER NOT NULL,
  merged_symbol TEXT NOT NULL
);

CREATE VIEW IF NOT EXISTS bpe_tokens AS
SELECT s.piece_id, s.position, v.token_id, s.symbol
FROM bpe_symbol AS s JOIN tokenizer_vocab AS v ON v.symbol = s.symbol;

CREATE TRIGGER IF NOT EXISTS initialize_bpe
AFTER INSERT ON bpe_piece
BEGIN
  INSERT INTO bpe_symbol(piece_id, position, symbol)
  WITH RECURSIVE positions(p) AS (
    SELECT 0 WHERE length(CAST(NEW.text AS BLOB)) > 0
    UNION ALL
    SELECT p + 1 FROM positions
    WHERE p + 1 < length(CAST(NEW.text AS BLOB))
  ), bytes AS (
    SELECT p, hex(substr(CAST(NEW.text AS BLOB), p + 1, 1)) AS h
    FROM positions
  )
  SELECT NEW.piece_id, p, bs.symbol
  FROM bytes AS b JOIN byte_symbol AS bs
    ON bs.byte_value =
      16 * (instr('0123456789ABCDEF', substr(b.h, 1, 1)) - 1)
           + (instr('0123456789ABCDEF', substr(b.h, 2, 1)) - 1);
  INSERT INTO bpe_step(piece_id, iteration)
  WITH RECURSIVE steps(n) AS (
    SELECT 0 WHERE length(CAST(NEW.text AS BLOB)) > 1
    UNION ALL SELECT n + 1 FROM steps
    WHERE n + 2 < length(CAST(NEW.text AS BLOB))
  )
  SELECT NEW.piece_id, n FROM steps ORDER BY n;
  SELECT RAISE(ABORT, 'recursive BPE triggers did not finish')
  WHERE EXISTS (
    SELECT 1 FROM (
      SELECT symbol AS left_symbol,
             lead(symbol) OVER (ORDER BY position) AS right_symbol
      FROM bpe_symbol WHERE piece_id = NEW.piece_id
    ) AS adjacent JOIN tokenizer_merge AS m
      ON m.left_symbol = adjacent.left_symbol
     AND m.right_symbol = adjacent.right_symbol
  );
END;

CREATE TRIGGER IF NOT EXISTS merge_bpe_pair
AFTER INSERT ON bpe_step
BEGIN
  DELETE FROM bpe_choice WHERE piece_id = NEW.piece_id;
  INSERT INTO bpe_choice(piece_id, left_position, right_position, merged_symbol)
  WITH adjacent AS (
    SELECT position AS left_position,
           lead(position) OVER (ORDER BY position) AS right_position,
           symbol AS left_symbol,
           lead(symbol) OVER (ORDER BY position) AS right_symbol
    FROM bpe_symbol WHERE piece_id = NEW.piece_id
  )
  SELECT NEW.piece_id, a.left_position, a.right_position,
         a.left_symbol || a.right_symbol
  FROM adjacent AS a JOIN tokenizer_merge AS m
    ON m.left_symbol = a.left_symbol AND m.right_symbol = a.right_symbol
  ORDER BY m.rank, a.left_position LIMIT 1;
  UPDATE bpe_symbol
  SET symbol = (SELECT merged_symbol FROM bpe_choice
                WHERE piece_id = NEW.piece_id)
  WHERE piece_id = NEW.piece_id AND position =
    (SELECT left_position FROM bpe_choice WHERE piece_id = NEW.piece_id);
  DELETE FROM bpe_symbol
  WHERE piece_id = NEW.piece_id AND position =
    (SELECT right_position FROM bpe_choice WHERE piece_id = NEW.piece_id);
END;
-- Decode an ordered token sequence through the tokenizer's byte alphabet.
CREATE TABLE IF NOT EXISTS decode_token (
  sequence_index INTEGER PRIMARY KEY CHECK (sequence_index >= 0),
  token_id INTEGER NOT NULL REFERENCES tokenizer_vocab(token_id)
);

CREATE VIEW IF NOT EXISTS decoded_text AS
WITH RECURSIVE tokens AS (
  SELECT d.sequence_index, v.symbol
  FROM decode_token AS d JOIN tokenizer_vocab AS v ON v.token_id = d.token_id
), chars(sequence_index, char_pos, symbol) AS (
  SELECT sequence_index, 0, symbol FROM tokens WHERE length(symbol) > 0
  UNION ALL
  SELECT sequence_index, char_pos + 1, symbol FROM chars
  WHERE char_pos + 1 < length(symbol)
), ordered AS (
  SELECT row_number() OVER (ORDER BY c.sequence_index, c.char_pos) AS n,
         printf('%02X', b.byte_value) AS hex_byte
  FROM chars AS c JOIN byte_symbol AS b
    ON b.symbol = substr(c.symbol, c.char_pos + 1, 1)
), assembled(n, hex_bytes) AS (
  SELECT 0, ''
  UNION ALL
  SELECT o.n, a.hex_bytes || o.hex_byte
  FROM assembled AS a JOIN ordered AS o ON o.n = a.n + 1
)
SELECT CAST(unhex(hex_bytes) AS TEXT) AS text
FROM assembled ORDER BY n DESC LIMIT 1;
-- Import Unicode General Category letter and number ranges as tokenizer data.
-- Source: https://www.unicode.org/Public/UCD/latest/ucd/UnicodeData.txt
CREATE TABLE IF NOT EXISTS unicode_range (
  start_cp INTEGER NOT NULL,
  end_cp INTEGER NOT NULL,
  class TEXT NOT NULL CHECK (class IN ('L','N')),
  PRIMARY KEY (start_cp, end_cp)
) WITHOUT ROWID;

INSERT OR REPLACE INTO unicode_range(start_cp, end_cp, class)
WITH raw_lines AS (
  SELECT CAST(j.key AS INTEGER) AS line_no, j.value AS line
  FROM json_each('["' ||
    replace(CAST(readfile('outputs/UnicodeData.txt') AS TEXT), char(10), '","')
    || '"]') AS j
  WHERE j.value <> ''
), fields AS (
  SELECT line_no,
    substr(line, 1, instr(line, ';') - 1) AS code_hex,
    substr(line, instr(line, ';') + 1) AS rest
  FROM raw_lines
), parsed AS (
  SELECT line_no, code_hex,
    substr(rest, 1, instr(rest, ';') - 1) AS name,
    substr(rest, instr(rest, ';') + 1, 2) AS category
  FROM fields
), digit_position(n) AS (
  VALUES (1),(2),(3),(4),(5),(6)
), codepoints AS (
  SELECT p.line_no, p.name, p.category,
    sum((instr('0123456789ABCDEF', substr(p.code_hex, d.n, 1)) - 1) *
        CAST(pow(16, length(p.code_hex) - d.n) AS INTEGER)) AS cp
  FROM parsed AS p JOIN digit_position AS d ON d.n <= length(p.code_hex)
  GROUP BY p.line_no
), with_next AS (
  SELECT cp, name, category, lead(cp) OVER (ORDER BY cp) AS next_cp
  FROM codepoints
)
SELECT cp, CASE WHEN name LIKE '%First>' THEN next_cp ELSE cp END,
       substr(category, 1, 1)
FROM with_next
WHERE substr(category, 1, 1) IN ('L','N')
  AND name NOT LIKE '%Last>';
-- GPT-2 style Unicode pretokenization, special tokens, and byte-level BPE.
-- Run 03/04 tokenizer import, 08_bpe.sql, and 10_unicode_data.sql first.
PRAGMA recursive_triggers = ON;
CREATE TABLE IF NOT EXISTS prompt_input (
  prompt_id INTEGER PRIMARY KEY CHECK (prompt_id BETWEEN 0 AND 1000000),
  text TEXT NOT NULL
);

CREATE VIEW IF NOT EXISTS prompt_segment AS
WITH RECURSIVE hits(prompt_id, text, token_id, content, start_pos) AS (
  SELECT p.prompt_id, p.text, s.token_id, s.content,
         instr(p.text, s.content) - 1
  FROM prompt_input AS p CROSS JOIN tokenizer_special AS s
  WHERE instr(p.text, s.content) > 0
  UNION ALL
  SELECT prompt_id, text, token_id, content,
         start_pos + length(content) +
           instr(substr(text, start_pos + length(content) + 1), content) - 1
  FROM hits
  WHERE instr(substr(text, start_pos + length(content) + 1), content) > 0
), ordered_hits AS (
  SELECT prompt_id, text, token_id, content, start_pos,
         coalesce(lag(start_pos + length(content)) OVER
           (PARTITION BY prompt_id ORDER BY start_pos), 0) AS prev_end
  FROM hits
), spans AS (
  SELECT prompt_id, prev_end AS start_pos,
         start_pos - prev_end AS span_length,
         NULL AS special_token_id
  FROM ordered_hits WHERE start_pos > prev_end
  UNION ALL
  SELECT prompt_id, start_pos, length(content), token_id
  FROM hits
  UNION ALL
  SELECT p.prompt_id,
         coalesce((SELECT max(h.start_pos + length(h.content))
                   FROM hits AS h WHERE h.prompt_id = p.prompt_id), 0),
         length(p.text) -
         coalesce((SELECT max(h.start_pos + length(h.content))
                   FROM hits AS h WHERE h.prompt_id = p.prompt_id), 0),
         NULL
  FROM prompt_input AS p
)
SELECT s.prompt_id,
       row_number() OVER (PARTITION BY s.prompt_id ORDER BY s.start_pos) - 1
         AS segment_index,
       s.start_pos,
       substr(p.text, s.start_pos + 1, s.span_length) AS text,
       s.special_token_id
FROM spans AS s JOIN prompt_input AS p ON p.prompt_id = s.prompt_id
WHERE s.span_length > 0;

CREATE VIEW IF NOT EXISTS prompt_piece AS
WITH RECURSIVE chars(prompt_id, segment_index, text, p, ch) AS (
  SELECT prompt_id, segment_index, text, 0, substr(text, 1, 1)
  FROM prompt_segment WHERE special_token_id IS NULL AND length(text) > 0
  UNION ALL
  SELECT prompt_id, segment_index, text, p + 1,
         substr(text, p + 2, 1)
  FROM chars WHERE p + 1 < length(text)
), classified AS (
  SELECT c.prompt_id, c.segment_index, c.text, c.p, c.ch,
    CASE WHEN x.start_pos IS NOT NULL THEN 'C'
         WHEN unicode(c.ch) BETWEEN 9 AND 13
           OR unicode(c.ch) BETWEEN 8192 AND 8202
           OR unicode(c.ch) IN
             (32,133,160,5760,8232,8233,8239,8287,12288) THEN 'W'
         WHEN u.end_cp >= unicode(c.ch) THEN u.class
         ELSE 'P' END AS class
  FROM chars AS c
  LEFT JOIN unicode_range AS u ON u.start_cp =
    (SELECT max(start_cp) FROM unicode_range
     WHERE start_cp <= unicode(c.ch))
  LEFT JOIN (
    SELECT prompt_id, segment_index, p AS start_pos,
      CASE WHEN substr(text, p + 1, 3) IN ('''re','''ve','''ll') THEN 3
           WHEN substr(text, p + 1, 2) IN ('''s','''t','''m','''d') THEN 2
           ELSE 0 END AS width
    FROM chars WHERE ch = ''''
  ) AS x ON x.prompt_id = c.prompt_id
    AND x.segment_index = c.segment_index
    AND c.p >= x.start_pos AND c.p < x.start_pos + x.width
), neighbors AS (
  SELECT *,
    lag(class) OVER
      (PARTITION BY prompt_id, segment_index ORDER BY p) AS prev_class,
    lead(class) OVER
      (PARTITION BY prompt_id, segment_index ORDER BY p) AS next_class,
    lag(ch) OVER
      (PARTITION BY prompt_id, segment_index ORDER BY p) AS prev_ch
  FROM classified
), boundaries AS (
  SELECT *, CASE
    WHEN p = 0 THEN 1
    WHEN class = 'W' AND prev_class = 'W'
       AND next_class IS NOT NULL AND next_class <> 'W' THEN 1
    WHEN class = 'W' AND prev_class <> 'W' THEN 1
    WHEN prev_ch = ' ' AND class IN ('L', 'P') THEN 0
    WHEN class <> prev_class THEN 1
    WHEN class = 'N' THEN 1
    ELSE 0 END AS starts_piece
  FROM neighbors
), numbered AS (
  SELECT *, sum(starts_piece) OVER
    (PARTITION BY prompt_id, segment_index ORDER BY p) AS piece_number
  FROM boundaries
), grouped AS (
  SELECT prompt_id, segment_index, piece_number,
         min(p) AS first_char, count(*) AS char_count, max(text) AS text
  FROM numbered GROUP BY prompt_id, segment_index, piece_number
)
SELECT prompt_id,
       segment_index * 1000000 + piece_number - 1 AS piece_index,
       substr(text, first_char + 1, char_count) AS piece
FROM grouped;

CREATE TRIGGER IF NOT EXISTS tokenize_prompt
AFTER INSERT ON prompt_input
BEGIN
  SELECT RAISE(ABORT, 'special-token segmentation mismatch')
  WHERE coalesce((SELECT sum(length(text)) FROM prompt_segment
                  WHERE prompt_id = NEW.prompt_id), 0) <> length(NEW.text);
  INSERT INTO bpe_piece(piece_id, text)
  SELECT NEW.prompt_id * 10000000000 + piece_index, piece
  FROM prompt_piece WHERE prompt_id = NEW.prompt_id
  ORDER BY piece_index;
  SELECT RAISE(ABORT, 'BPE produced a symbol outside the vocabulary')
  WHERE (SELECT count(*) FROM bpe_symbol
         WHERE piece_id BETWEEN NEW.prompt_id * 10000000000
             AND NEW.prompt_id * 10000000000 + 9999999999)
       <> (SELECT count(*) FROM bpe_tokens
           WHERE piece_id BETWEEN NEW.prompt_id * 10000000000
               AND NEW.prompt_id * 10000000000 + 9999999999);
END;

CREATE VIEW IF NOT EXISTS prompt_tokens AS
SELECT p.prompt_id, p.piece_index, t.position, t.token_id, t.symbol
FROM prompt_piece AS p JOIN bpe_tokens AS t
  ON t.piece_id = p.prompt_id * 10000000000 + p.piece_index
UNION ALL
SELECT s.prompt_id, s.segment_index * 1000000, 0,
       s.special_token_id, v.symbol
FROM prompt_segment AS s JOIN tokenizer_vocab AS v
  ON v.token_id = s.special_token_id
WHERE s.special_token_id IS NOT NULL;
-- Mark where the prompt ends before inserting its token IDs into inference_input.
CREATE TABLE IF NOT EXISTS generation_config (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  prompt_length INTEGER NOT NULL CHECK (prompt_length BETWEEN 1 AND 2048)
);
INSERT OR IGNORE INTO generation_config VALUES (1, 1);

-- Predictions from intermediate prompt positions are excluded from output.
CREATE TRIGGER IF NOT EXISTS decode_generated_token
AFTER INSERT ON generated
BEGIN
  INSERT OR REPLACE INTO decode_token(sequence_index, token_id)
  SELECT NEW.pos - prompt_length + 1, NEW.next_token_id
  FROM generation_config
  WHERE id = 1 AND NEW.pos >= prompt_length - 1
    AND NEW.next_token_id <> 0;
END;

CREATE VIEW IF NOT EXISTS generated_text AS
SELECT text FROM decoded_text;
-- After prompt IDs have been inserted, one generation_request advances the
-- model for any number of additional positions, entirely through SQL triggers.
-- For N output tokens, request N-1 additional steps; the last prompt position
-- has already produced the first output token.
PRAGMA recursive_triggers = ON;
CREATE TABLE IF NOT EXISTS generation_step (
  step_index INTEGER PRIMARY KEY CHECK (step_index >= 0)
);
CREATE TABLE IF NOT EXISTS generation_request (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  additional_steps INTEGER NOT NULL CHECK (additional_steps BETWEEN 0 AND 2047)
);

CREATE TRIGGER IF NOT EXISTS run_generation_step
AFTER INSERT ON generation_step
BEGIN
  SELECT RAISE(ABORT, 'insert prompt tokens before generating')
  WHERE NOT EXISTS (SELECT 1 FROM generated);
  INSERT INTO inference_input(pos, token_id)
  SELECT pos + 1, next_token_id
  FROM (SELECT pos, next_token_id FROM generated
        ORDER BY pos DESC LIMIT 1)
  WHERE next_token_id <> 0;
END;

CREATE TRIGGER IF NOT EXISTS run_generation_request
AFTER INSERT ON generation_request
BEGIN
  DELETE FROM generation_step;
  INSERT INTO generation_step(step_index)
  WITH RECURSIVE numbers(n) AS (
    SELECT 0 WHERE NEW.additional_steps > 0
    UNION ALL
    SELECT n + 1 FROM numbers WHERE n + 1 < NEW.additional_steps
  )
  SELECT n FROM numbers;
END;
-- Run a freshly tokenized ASCII prompt by inserting its prompt_id here.
-- This advances the model once for each prompt token, in token order.
PRAGMA recursive_triggers = ON;
CREATE TABLE IF NOT EXISTS prompt_run (
  prompt_id INTEGER PRIMARY KEY REFERENCES prompt_input(prompt_id)
);

CREATE TRIGGER IF NOT EXISTS run_prompt
AFTER INSERT ON prompt_run
BEGIN
  SELECT RAISE(ABORT, 'prompt must encode to 1..2048 tokens')
  WHERE (SELECT count(*) FROM prompt_tokens
         WHERE prompt_id = NEW.prompt_id) NOT BETWEEN 1 AND 2048;
  SELECT RAISE(ABORT, 'reset inference state before a new prompt')
  WHERE EXISTS (SELECT 1 FROM inference_input);
  UPDATE generation_config SET prompt_length =
    (SELECT count(*) FROM prompt_tokens
     WHERE prompt_id = NEW.prompt_id) WHERE id = 1;
  INSERT INTO inference_input(pos, token_id)
  SELECT row_number() OVER (ORDER BY piece_index, position) - 1,
         token_id
  FROM prompt_tokens WHERE prompt_id = NEW.prompt_id
  ORDER BY piece_index, position;
END;
