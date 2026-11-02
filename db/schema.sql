CREATE TABLE IF NOT EXISTS pitchers (
    pitcher_id  BIGINT PRIMARY KEY,
    name        TEXT   NOT NULL,
    throws      CHAR(1) NOT NULL CHECK (throws IN ('L', 'R'))
);

CREATE TABLE IF NOT EXISTS games (
    game_pk    BIGINT   PRIMARY KEY,
    game_date  DATE     NOT NULL,
    season     SMALLINT NOT NULL,
    home_team  TEXT     NOT NULL,
    away_team  TEXT     NOT NULL
);

CREATE TABLE IF NOT EXISTS pitches (
    game_pk        BIGINT   NOT NULL REFERENCES games (game_pk),
    at_bat_number  INTEGER  NOT NULL CHECK (at_bat_number >= 1),
    pitch_number   INTEGER  NOT NULL CHECK (pitch_number >= 1),
    pitcher_id     BIGINT   NOT NULL REFERENCES pitchers (pitcher_id),
    batter_id      BIGINT   NOT NULL,
    stand          CHAR(1)  NOT NULL CHECK (stand IN ('L', 'R')),
    inning         SMALLINT NOT NULL CHECK (inning >= 1),
    inning_topbot  TEXT     NOT NULL CHECK (inning_topbot IN ('Top', 'Bot')),
    balls          SMALLINT NOT NULL CHECK (balls BETWEEN 0 AND 3),
    strikes        SMALLINT NOT NULL CHECK (strikes BETWEEN 0 AND 2),
    outs_when_up   SMALLINT NOT NULL CHECK (outs_when_up BETWEEN 0 AND 2),
    on_1b          BIGINT,           -- runner's player id, NULL = base empty
    on_2b          BIGINT,
    on_3b          BIGINT,
    home_score     SMALLINT NOT NULL,
    away_score     SMALLINT NOT NULL,
    bat_score      SMALLINT NOT NULL,
    fld_score      SMALLINT NOT NULL,
    pitch_type     TEXT     NOT NULL,  -- Statcast's code for the pitch as recorded
    pitch_class    TEXT     NOT NULL CHECK (pitch_class IN
                       ('FF', 'SI', 'FC', 'SL', 'ST', 'CU', 'KC', 'CH', 'FS', 'SV')),
    description    TEXT     NOT NULL,  
    type           CHAR(1),
    n_thruorder_pitcher SMALLINT,

    PRIMARY KEY (game_pk, at_bat_number, pitch_number)
);

-- The primary key already starts with game_pk, so it serves "all pitches of a game".
CREATE INDEX IF NOT EXISTS pitches_by_pitcher
    ON pitches (pitcher_id, game_pk, at_bat_number, pitch_number);