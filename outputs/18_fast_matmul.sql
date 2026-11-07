-- Replace the raw-byte matmul trigger after 17_activate_real.sql.
-- A range scan per output row avoids a temporary GROUP BY B-tree.
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
  WITH RECURSIVE spec AS (
    SELECT id.tensor_id, json_extract(t.shape, '$[0]') AS output_rows,
           json_extract(t.shape, '$[1]') AS cols
    FROM tensor_identity AS id JOIN tensor AS t ON t.name = id.name
    WHERE id.name = NEW.tensor_name
  ), rows(r) AS (
    SELECT NEW.first_row
    UNION ALL
    SELECT r + 1 FROM rows
    WHERE r + 1 < coalesce(NEW.end_row,
      (SELECT output_rows FROM spec))
  )
  SELECT r,
    (SELECT sum(w.value * i.value)
     FROM weight_real AS w
     JOIN matrix_input AS i ON i.dim = w.linear_index - r * spec.cols
     WHERE w.tensor_id = spec.tensor_id
       AND w.linear_index >= r * spec.cols
       AND w.linear_index < (r + 1) * spec.cols)
  FROM rows CROSS JOIN spec;
  SELECT RAISE(ABORT, 'matrix output row mismatch')
  WHERE (SELECT count(*) FROM matrix_output) <>
    (coalesce(NEW.end_row,
      (SELECT json_extract(shape, '$[0]') FROM tensor
       WHERE name = NEW.tensor_name)) - NEW.first_row);
END;
