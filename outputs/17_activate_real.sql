-- Switch scalar reads to predecoded REAL rows only after complete materialization.
CREATE TEMP TABLE IF NOT EXISTS real_weight_guard (
  ok INTEGER NOT NULL CHECK (ok = 1)
);
INSERT OR ROLLBACK INTO real_weight_guard(ok)
SELECT CASE WHEN (SELECT count(*) FROM weight_real) =
                 (SELECT sum(elements) FROM tensor)
             AND NOT EXISTS (
               SELECT 1 FROM tensor_identity AS i
               JOIN tensor AS t ON t.name = i.name
               LEFT JOIN weight_real AS w ON w.tensor_id = i.tensor_id
               GROUP BY i.tensor_id
               HAVING count(w.linear_index) <> t.elements
                   OR min(w.linear_index) <> 0
                   OR max(w.linear_index) <> t.elements - 1
             )
            THEN 1 ELSE 0 END;
DROP VIEW scalar_value;
CREATE VIEW scalar_value AS
SELECT r.name, r.linear_index, w.value
FROM scalar_request AS r
JOIN tensor_identity AS t ON t.name = r.name
JOIN weight_real AS w
  ON w.tensor_id = t.tensor_id AND w.linear_index = r.linear_index;
