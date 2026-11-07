-- Optional acceleration: decode F32/BF16/F16 weights once into REAL rows.
-- Run 11_row_shards.sql before this file, then insert materialize_job rows.
-- After all weights are decoded, run 17_activate_real.sql.
CREATE TABLE IF NOT EXISTS tensor_identity (
  tensor_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL UNIQUE REFERENCES tensor(name)
);
INSERT OR IGNORE INTO tensor_identity(tensor_id, name)
SELECT row_number() OVER (ORDER BY name), name FROM tensor;

CREATE TABLE IF NOT EXISTS weight_real (
  tensor_id INTEGER NOT NULL,
  linear_index INTEGER NOT NULL CHECK (linear_index >= 0),
  value REAL NOT NULL,
  PRIMARY KEY (tensor_id, linear_index)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS materialize_job (
  name TEXT NOT NULL REFERENCES tensor(name),
  first_row INTEGER NOT NULL,
  end_row INTEGER NOT NULL CHECK (end_row > first_row),
  PRIMARY KEY (name, first_row)
) WITHOUT ROWID;

CREATE TRIGGER IF NOT EXISTS run_materialize_job
AFTER INSERT ON materialize_job
BEGIN
  DELETE FROM scalar_request;
  INSERT INTO scalar_request(name, linear_index)
  WITH RECURSIVE rows(r) AS (
    SELECT NEW.first_row
    UNION ALL SELECT r + 1 FROM rows WHERE r + 1 < NEW.end_row
  ), cols(c) AS (
    SELECT 0
    UNION ALL SELECT c + 1 FROM cols
    WHERE c + 1 < (SELECT CASE WHEN json_array_length(shape) = 2
                         THEN json_extract(shape, '$[1]') ELSE elements END
                   FROM tensor WHERE name = NEW.name)
  )
  SELECT NEW.name,
         r * CASE WHEN json_array_length(t.shape) = 2
             THEN json_extract(t.shape, '$[1]') ELSE t.elements END + c
  FROM tensor AS t CROSS JOIN rows CROSS JOIN cols
  WHERE t.name = NEW.name;
  INSERT OR REPLACE INTO weight_real(tensor_id, linear_index, value)
  SELECT i.tensor_id, s.linear_index, s.value
  FROM scalar_value AS s JOIN tensor_identity AS i ON i.name = s.name
  WHERE s.name = NEW.name;
  DELETE FROM scalar_request;
END;
