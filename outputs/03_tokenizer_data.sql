-- Import the published tokenizer.json entirely with SQL.
-- Example: INSERT INTO tokenizer_file VALUES (1, readfile('work/tokenizer.json'));
CREATE TABLE IF NOT EXISTS tokenizer_file (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  bytes BLOB NOT NULL CHECK (typeof(bytes) = 'blob')
);

CREATE TABLE IF NOT EXISTS tokenizer_vocab (
  symbol TEXT PRIMARY KEY,
  token_id INTEGER NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS tokenizer_merge (
  rank INTEGER PRIMARY KEY,
  left_symbol TEXT NOT NULL,
  right_symbol TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS tokenizer_merge_pair
ON tokenizer_merge(left_symbol, right_symbol);

CREATE TABLE IF NOT EXISTS tokenizer_special (
  token_id INTEGER PRIMARY KEY,
  content TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS byte_symbol (
  byte_value INTEGER PRIMARY KEY CHECK (byte_value BETWEEN 0 AND 255),
  symbol TEXT NOT NULL UNIQUE
);

WITH RECURSIVE byte_range(b) AS (
  SELECT 0 UNION ALL SELECT b + 1 FROM byte_range WHERE b < 255
), tagged AS (
  SELECT b,
    CASE WHEN b BETWEEN 33 AND 126 OR b BETWEEN 161 AND 172
              OR b BETWEEN 174 AND 255 THEN 1 ELSE 0 END AS direct
  FROM byte_range
), numbered AS (
  SELECT b, direct,
    sum(CASE WHEN direct = 0 THEN 1 ELSE 0 END)
      OVER (ORDER BY b) - 1 AS extra_index
  FROM tagged
)
INSERT OR REPLACE INTO byte_symbol(byte_value, symbol)
SELECT b, char(CASE WHEN direct = 1 THEN b ELSE 256 + extra_index END)
FROM numbered;

