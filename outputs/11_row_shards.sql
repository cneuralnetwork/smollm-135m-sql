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
