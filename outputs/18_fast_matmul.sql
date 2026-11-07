-- Replace the raw-byte matmul trigger after 17_activate_real.sql.
DROP TRIGGER IF EXISTS run_matrix_job;
CREATE TRIGGER run_matrix_job
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
  DELETE FROM matrix_output;
  INSERT INTO matrix_output(dim, value)
  SELECT CAST(w.linear_index / json_extract(t.shape, '$[1]') AS INTEGER),
         sum(w.value * i.value)
  FROM tensor_identity AS id
  JOIN tensor AS t ON t.name = id.name
  JOIN weight_real AS w
    ON w.tensor_id = id.tensor_id
   AND w.linear_index >= NEW.first_row * json_extract(t.shape, '$[1]')
   AND w.linear_index < coalesce(NEW.end_row,
       json_extract(t.shape, '$[0]')) * json_extract(t.shape, '$[1]')
  JOIN matrix_input AS i
    ON i.dim = w.linear_index % json_extract(t.shape, '$[1]')
  WHERE id.name = NEW.tensor_name
  GROUP BY 1;
  SELECT RAISE(ABORT, 'matrix output row mismatch')
  WHERE (SELECT count(*) FROM matrix_output) <>
    (coalesce(NEW.end_row,
      (SELECT json_extract(shape, '$[0]') FROM tensor
       WHERE name = NEW.tensor_name)) - NEW.first_row);
END;
