-- Run after inserting tokenizer_file(1, readfile('outputs/tokenizer.json')).
BEGIN;
INSERT OR REPLACE INTO tokenizer_vocab(symbol, token_id)
SELECT j.key, CAST(j.value AS INTEGER)
FROM tokenizer_file AS f, json_each(CAST(f.bytes AS TEXT), '$.model.vocab') AS j
WHERE f.id = 1;

INSERT OR REPLACE INTO tokenizer_merge(rank, left_symbol, right_symbol)
SELECT CAST(j.key AS INTEGER),
       substr(j.value, 1, instr(j.value, ' ') - 1),
       substr(j.value, instr(j.value, ' ') + 1)
FROM tokenizer_file AS f, json_each(CAST(f.bytes AS TEXT), '$.model.merges') AS j
WHERE f.id = 1 AND instr(j.value, ' ') > 0;
INSERT OR REPLACE INTO tokenizer_special(token_id, content)
SELECT json_extract(j.value, '$.id'), json_extract(j.value, '$.content')
FROM tokenizer_file AS f, json_each(CAST(f.bytes AS TEXT), '$.added_tokens') AS j
WHERE f.id = 1 AND json_extract(j.value, '$.special') = 1;
DELETE FROM tokenizer_file;
COMMIT;
