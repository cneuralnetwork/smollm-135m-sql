-- Byte-level BPE on one already pretokenized piece. SQLite SQL only.
-- Insert (piece_id, text) into bpe_piece, then read bpe_tokens.
PRAGMA recursive_triggers = ON;
CREATE TABLE IF NOT EXISTS bpe_piece (
  piece_id INTEGER PRIMARY KEY,
  text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS bpe_symbol (
  piece_id INTEGER NOT NULL,
  position INTEGER NOT NULL,
  symbol TEXT NOT NULL,
  PRIMARY KEY (piece_id, position)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS bpe_step (
  piece_id INTEGER NOT NULL,
  iteration INTEGER NOT NULL,
  PRIMARY KEY (piece_id, iteration)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS bpe_choice (
  piece_id INTEGER PRIMARY KEY,
  left_position INTEGER NOT NULL,
  right_position INTEGER NOT NULL,
  merged_symbol TEXT NOT NULL
);

CREATE VIEW IF NOT EXISTS bpe_tokens AS
SELECT s.piece_id, s.position, v.token_id, s.symbol
FROM bpe_symbol AS s JOIN tokenizer_vocab AS v ON v.symbol = s.symbol;

CREATE TRIGGER IF NOT EXISTS initialize_bpe
AFTER INSERT ON bpe_piece
BEGIN
  INSERT INTO bpe_symbol(piece_id, position, symbol)
  WITH RECURSIVE positions(p) AS (
    SELECT 0 WHERE length(CAST(NEW.text AS BLOB)) > 0
    UNION ALL
    SELECT p + 1 FROM positions
    WHERE p + 1 < length(CAST(NEW.text AS BLOB))
  ), bytes AS (
    SELECT p, hex(substr(CAST(NEW.text AS BLOB), p + 1, 1)) AS h
    FROM positions
  )
  SELECT NEW.piece_id, p, bs.symbol
  FROM bytes AS b JOIN byte_symbol AS bs
    ON bs.byte_value =
      16 * (instr('0123456789ABCDEF', substr(b.h, 1, 1)) - 1)
           + (instr('0123456789ABCDEF', substr(b.h, 2, 1)) - 1);
  INSERT INTO bpe_step(piece_id, iteration)
  WITH RECURSIVE steps(n) AS (
    SELECT 0 WHERE length(CAST(NEW.text AS BLOB)) > 1
    UNION ALL SELECT n + 1 FROM steps
    WHERE n + 2 < length(CAST(NEW.text AS BLOB))
  )
  SELECT NEW.piece_id, n FROM steps ORDER BY n;
  SELECT RAISE(ABORT, 'recursive BPE triggers did not finish')
  WHERE EXISTS (
    SELECT 1 FROM (
      SELECT symbol AS left_symbol,
             lead(symbol) OVER (ORDER BY position) AS right_symbol
      FROM bpe_symbol WHERE piece_id = NEW.piece_id
    ) AS adjacent JOIN tokenizer_merge AS m
      ON m.left_symbol = adjacent.left_symbol
     AND m.right_symbol = adjacent.right_symbol
  );
END;

CREATE TRIGGER IF NOT EXISTS merge_bpe_pair
AFTER INSERT ON bpe_step
BEGIN
  DELETE FROM bpe_choice WHERE piece_id = NEW.piece_id;
  INSERT INTO bpe_choice(piece_id, left_position, right_position, merged_symbol)
  WITH adjacent AS (
    SELECT position AS left_position,
           lead(position) OVER (ORDER BY position) AS right_position,
           symbol AS left_symbol,
           lead(symbol) OVER (ORDER BY position) AS right_symbol
    FROM bpe_symbol WHERE piece_id = NEW.piece_id
  )
  SELECT NEW.piece_id, a.left_position, a.right_position,
         a.left_symbol || a.right_symbol
  FROM adjacent AS a JOIN tokenizer_merge AS m
    ON m.left_symbol = a.left_symbol AND m.right_symbol = a.right_symbol
  ORDER BY m.rank, a.left_position LIMIT 1;
  UPDATE bpe_symbol
  SET symbol = (SELECT merged_symbol FROM bpe_choice
                WHERE piece_id = NEW.piece_id)
  WHERE piece_id = NEW.piece_id AND position =
    (SELECT left_position FROM bpe_choice WHERE piece_id = NEW.piece_id);
  DELETE FROM bpe_symbol
  WHERE piece_id = NEW.piece_id AND position =
    (SELECT right_position FROM bpe_choice WHERE piece_id = NEW.piece_id);
END;
