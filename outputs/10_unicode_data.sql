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
