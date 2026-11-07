-- GPT-2 style Unicode pretokenization, special tokens, and byte-level BPE.
-- Run 03/04 tokenizer import, 08_bpe.sql, and 10_unicode_data.sql first.
PRAGMA recursive_triggers = ON;
CREATE TABLE IF NOT EXISTS prompt_input (
  prompt_id INTEGER PRIMARY KEY CHECK (prompt_id BETWEEN 0 AND 1000000),
  text TEXT NOT NULL
);

CREATE VIEW IF NOT EXISTS prompt_segment AS
WITH RECURSIVE hits(prompt_id, text, token_id, content, start_pos) AS (
  SELECT p.prompt_id, p.text, s.token_id, s.content,
         instr(p.text, s.content) - 1
  FROM prompt_input AS p CROSS JOIN tokenizer_special AS s
  WHERE instr(p.text, s.content) > 0
  UNION ALL
  SELECT prompt_id, text, token_id, content,
         start_pos + length(content) +
           instr(substr(text, start_pos + length(content) + 1), content) - 1
  FROM hits
  WHERE instr(substr(text, start_pos + length(content) + 1), content) > 0
), ordered_hits AS (
  SELECT prompt_id, text, token_id, content, start_pos,
         coalesce(lag(start_pos + length(content)) OVER
           (PARTITION BY prompt_id ORDER BY start_pos), 0) AS prev_end
  FROM hits
), spans AS (
  SELECT prompt_id, prev_end AS start_pos,
         start_pos - prev_end AS span_length,
         NULL AS special_token_id
  FROM ordered_hits WHERE start_pos > prev_end
  UNION ALL
  SELECT prompt_id, start_pos, length(content), token_id
  FROM hits
  UNION ALL
  SELECT p.prompt_id,
         coalesce((SELECT max(h.start_pos + length(h.content))
                   FROM hits AS h WHERE h.prompt_id = p.prompt_id), 0),
         length(p.text) -
         coalesce((SELECT max(h.start_pos + length(h.content))
                   FROM hits AS h WHERE h.prompt_id = p.prompt_id), 0),
         NULL
  FROM prompt_input AS p
)
SELECT s.prompt_id,
       row_number() OVER (PARTITION BY s.prompt_id ORDER BY s.start_pos) - 1
         AS segment_index,
       s.start_pos,
       substr(p.text, s.start_pos + 1, s.span_length) AS text,
       s.special_token_id
FROM spans AS s JOIN prompt_input AS p ON p.prompt_id = s.prompt_id
WHERE s.span_length > 0;

CREATE VIEW IF NOT EXISTS prompt_piece AS
WITH RECURSIVE chars(prompt_id, segment_index, text, p, ch) AS (
  SELECT prompt_id, segment_index, text, 0, substr(text, 1, 1)
  FROM prompt_segment WHERE special_token_id IS NULL AND length(text) > 0
  UNION ALL
  SELECT prompt_id, segment_index, text, p + 1,
         substr(text, p + 2, 1)
  FROM chars WHERE p + 1 < length(text)
), classified AS (
  SELECT c.prompt_id, c.segment_index, c.text, c.p, c.ch,
    CASE WHEN x.start_pos IS NOT NULL THEN 'C'
         WHEN unicode(c.ch) BETWEEN 9 AND 13
           OR unicode(c.ch) BETWEEN 8192 AND 8202
           OR unicode(c.ch) IN
             (32,133,160,5760,8232,8233,8239,8287,12288) THEN 'W'
         WHEN u.end_cp >= unicode(c.ch) THEN u.class
         ELSE 'P' END AS class
  FROM chars AS c
  LEFT JOIN unicode_range AS u ON u.start_cp =
    (SELECT max(start_cp) FROM unicode_range
     WHERE start_cp <= unicode(c.ch))
  LEFT JOIN (
    SELECT prompt_id, segment_index, p AS start_pos,
      CASE WHEN substr(text, p + 1, 3) IN ('''re','''ve','''ll') THEN 3
           WHEN substr(text, p + 1, 2) IN ('''s','''t','''m','''d') THEN 2
           ELSE 0 END AS width
    FROM chars WHERE ch = ''''
  ) AS x ON x.prompt_id = c.prompt_id
    AND x.segment_index = c.segment_index
    AND c.p >= x.start_pos AND c.p < x.start_pos + x.width
), neighbors AS (
  SELECT *,
    lag(class) OVER
      (PARTITION BY prompt_id, segment_index ORDER BY p) AS prev_class,
    lead(class) OVER
      (PARTITION BY prompt_id, segment_index ORDER BY p) AS next_class,
    lag(ch) OVER
      (PARTITION BY prompt_id, segment_index ORDER BY p) AS prev_ch
  FROM classified
), boundaries AS (
  SELECT *, CASE
    WHEN p = 0 THEN 1
    WHEN class = 'W' AND prev_class = 'W'
       AND next_class IS NOT NULL AND next_class <> 'W' THEN 1
    WHEN class = 'W' AND prev_class <> 'W' THEN 1
    WHEN prev_ch = ' ' AND class IN ('L', 'P') THEN 0
    WHEN class <> prev_class THEN 1
    WHEN class = 'N' THEN 1
    ELSE 0 END AS starts_piece
  FROM neighbors
), numbered AS (
  SELECT *, sum(starts_piece) OVER
    (PARTITION BY prompt_id, segment_index ORDER BY p) AS piece_number
  FROM boundaries
), grouped AS (
  SELECT prompt_id, segment_index, piece_number,
         min(p) AS first_char, count(*) AS char_count, max(text) AS text
  FROM numbered GROUP BY prompt_id, segment_index, piece_number
)
SELECT prompt_id,
       segment_index * 1000000 + piece_number - 1 AS piece_index,
       substr(text, first_char + 1, char_count) AS piece
FROM grouped;

CREATE TRIGGER IF NOT EXISTS tokenize_prompt
AFTER INSERT ON prompt_input
BEGIN
  SELECT RAISE(ABORT, 'special-token segmentation mismatch')
  WHERE coalesce((SELECT sum(length(text)) FROM prompt_segment
                  WHERE prompt_id = NEW.prompt_id), 0) <> length(NEW.text);
  INSERT INTO bpe_piece(piece_id, text)
  SELECT NEW.prompt_id * 10000000000 + piece_index, piece
  FROM prompt_piece WHERE prompt_id = NEW.prompt_id
  ORDER BY piece_index;
  SELECT RAISE(ABORT, 'BPE produced a symbol outside the vocabulary')
  WHERE (SELECT count(*) FROM bpe_symbol
         WHERE piece_id BETWEEN NEW.prompt_id * 10000000000
             AND NEW.prompt_id * 10000000000 + 9999999999)
       <> (SELECT count(*) FROM bpe_tokens
           WHERE piece_id BETWEEN NEW.prompt_id * 10000000000
               AND NEW.prompt_id * 10000000000 + 9999999999);
END;

CREATE VIEW IF NOT EXISTS prompt_tokens AS
SELECT p.prompt_id, p.piece_index, t.position, t.token_id, t.symbol
FROM prompt_piece AS p JOIN bpe_tokens AS t
  ON t.piece_id = p.prompt_id * 10000000000 + p.piece_index
UNION ALL
SELECT s.prompt_id, s.segment_index * 1000000, 0,
       s.special_token_id, v.symbol
FROM prompt_segment AS s JOIN tokenizer_vocab AS v
  ON v.token_id = s.special_token_id
WHERE s.special_token_id IS NOT NULL;
