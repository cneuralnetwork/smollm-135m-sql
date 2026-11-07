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
