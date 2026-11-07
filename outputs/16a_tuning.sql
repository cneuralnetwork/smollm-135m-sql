-- Optional one-time import speed settings. Resume is safe via INSERT OR IGNORE.
PRAGMA mmap_size = 1073741824;
PRAGMA cache_size = -100000;
PRAGMA journal_mode = OFF;
PRAGMA synchronous = OFF;
