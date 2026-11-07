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
