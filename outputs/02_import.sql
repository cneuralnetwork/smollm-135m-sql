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
