-- Mark where the prompt ends before inserting its token IDs into inference_input.
CREATE TABLE IF NOT EXISTS generation_config (
  id INTEGER PRIMARY KEY CHECK (id = 1),
  prompt_length INTEGER NOT NULL CHECK (prompt_length BETWEEN 1 AND 2048)
);
INSERT OR IGNORE INTO generation_config VALUES (1, 1);

-- Predictions from intermediate prompt positions are excluded from output.
CREATE TRIGGER IF NOT EXISTS decode_generated_token
AFTER INSERT ON generated
BEGIN
  INSERT OR REPLACE INTO decode_token(sequence_index, token_id)
  SELECT NEW.pos - prompt_length + 1, NEW.next_token_id
  FROM generation_config
  WHERE id = 1 AND NEW.pos >= prompt_length - 1
    AND NEW.next_token_id <> 0;
END;

CREATE VIEW IF NOT EXISTS generated_text AS
SELECT text FROM decoded_text;
