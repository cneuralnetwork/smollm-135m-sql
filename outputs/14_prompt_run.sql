-- Run a freshly tokenized ASCII prompt by inserting its prompt_id here.
-- This advances the model once for each prompt token, in token order.
PRAGMA recursive_triggers = ON;
CREATE TABLE IF NOT EXISTS prompt_run (
  prompt_id INTEGER PRIMARY KEY REFERENCES prompt_input(prompt_id)
);

CREATE TRIGGER IF NOT EXISTS run_prompt
AFTER INSERT ON prompt_run
BEGIN
  SELECT RAISE(ABORT, 'prompt must encode to 1..2048 tokens')
  WHERE (SELECT count(*) FROM prompt_tokens
         WHERE prompt_id = NEW.prompt_id) NOT BETWEEN 1 AND 2048;
  SELECT RAISE(ABORT, 'reset inference state before a new prompt')
  WHERE EXISTS (SELECT 1 FROM inference_input);
  UPDATE generation_config SET prompt_length =
    (SELECT count(*) FROM prompt_tokens
     WHERE prompt_id = NEW.prompt_id) WHERE id = 1;
  INSERT INTO inference_input(pos, token_id)
  SELECT row_number() OVER (ORDER BY piece_index, position) - 1,
         token_id
  FROM prompt_tokens WHERE prompt_id = NEW.prompt_id
  ORDER BY piece_index, position;
END;
