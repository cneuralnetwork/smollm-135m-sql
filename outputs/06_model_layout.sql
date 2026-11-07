-- Expected HuggingFaceTB/SmolLM-135M tensor layout (config.json).
CREATE VIEW IF NOT EXISTS expected_tensor AS
WITH RECURSIVE layer(n) AS (
  SELECT 0 UNION ALL SELECT n + 1 FROM layer WHERE n < 29
), suffix(s, shape) AS (
  VALUES
    ('input_layernorm.weight', '[576]'),
    ('post_attention_layernorm.weight', '[576]'),
    ('self_attn.q_proj.weight', '[576,576]'),
    ('self_attn.k_proj.weight', '[192,576]'),
    ('self_attn.v_proj.weight', '[192,576]'),
    ('self_attn.o_proj.weight', '[576,576]'),
    ('mlp.gate_proj.weight', '[1536,576]'),
    ('mlp.up_proj.weight', '[1536,576]'),
    ('mlp.down_proj.weight', '[576,1536]')
)
SELECT 'model.embed_tokens.weight' AS name, '[49152,576]' AS shape
UNION ALL SELECT 'model.norm.weight', '[576]'
UNION ALL SELECT 'model.layers.' || layer.n || '.' || suffix.s, suffix.shape
FROM layer CROSS JOIN suffix;

CREATE VIEW IF NOT EXISTS model_problem AS
SELECT e.name, 'missing or wrong shape/dtype' AS problem
FROM expected_tensor AS e
LEFT JOIN tensor AS t ON t.name = e.name
WHERE t.name IS NULL OR t.shape <> e.shape OR t.dtype <> 'F32'
UNION ALL
SELECT t.name, 'unexpected tensor' FROM tensor AS t
LEFT JOIN expected_tensor AS e ON e.name = t.name
WHERE e.name IS NULL;
