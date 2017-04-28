CREATE TABLE mydb.users
(
  user_id UInt64,
  locale String,
  birthyear Float32,
  gender String,
  joinedAt String,
  location String,
  timezone Float32,
  count Float32,
  lat Float32,
  lng Float32
) ENGINE = Log;
