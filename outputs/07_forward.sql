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
