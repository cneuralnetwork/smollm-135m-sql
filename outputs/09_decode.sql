-- Decode an ordered token sequence through the tokenizer's byte alphabet.
CREATE TABLE IF NOT EXISTS decode_token (
  sequence_index INTEGER PRIMARY KEY CHECK (sequence_index >= 0),
  token_id INTEGER NOT NULL REFERENCES tokenizer_vocab(token_id)
);

CREATE VIEW IF NOT EXISTS decoded_text AS
WITH RECURSIVE tokens AS (
  SELECT d.sequence_index, v.symbol
  FROM decode_token AS d JOIN tokenizer_vocab AS v ON v.token_id = d.token_id
), chars(sequence_index, char_pos, symbol) AS (
  SELECT sequence_index, 0, symbol FROM tokens WHERE length(symbol) > 0
  UNION ALL
  SELECT sequence_index, char_pos + 1, symbol FROM chars
  WHERE char_pos + 1 < length(symbol)
), ordered AS (
  SELECT row_number() OVER (ORDER BY c.sequence_index, c.char_pos) AS n,
         printf('%02X', b.byte_value) AS hex_byte
  FROM chars AS c JOIN byte_symbol AS b
    ON b.symbol = substr(c.symbol, c.char_pos + 1, 1)
), assembled(n, hex_bytes) AS (
  SELECT 0, ''
  UNION ALL
  SELECT o.n, a.hex_bytes || o.hex_byte
  FROM assembled AS a JOIN ordered AS o ON o.n = a.n + 1
)
SELECT CAST(unhex(hex_bytes) AS TEXT) AS text
FROM assembled ORDER BY n DESC LIMIT 1;
