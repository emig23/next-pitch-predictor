-- name: pitch_mix_by_count
-- How often does one pitcher throw each pitch in each count?
SELECT pi.name, p.balls, p.strikes, p.pitch_class,
       count(*) AS pitches,
       round(100.0 * count(*) / sum(count(*)) OVER (PARTITION BY p.balls, p.strikes), 1) AS pct_of_count
FROM pitches p
JOIN pitchers pi ON pi.pitcher_id = p.pitcher_id
WHERE pi.name = 'Wheeler, Zack'
GROUP BY pi.name, p.balls, p.strikes, p.pitch_class
ORDER BY p.balls, p.strikes, pitches DESC;

-- name: pitches_per_game
-- Every game one pitcher appeared in, with date, teams and number of pitches (joins all 3 tables).
SELECT g.game_date, g.home_team, g.away_team, pi.name, count(*) AS pitches
FROM pitches p
JOIN games g     ON g.game_pk = p.game_pk
JOIN pitchers pi ON pi.pitcher_id = p.pitcher_id
WHERE pi.name = 'Wheeler, Zack'
GROUP BY g.game_date, g.home_team, g.away_team, pi.name
ORDER BY g.game_date;

-- name: favorite_pitch_by_season
-- Each pitcher's most-used pitch in each season, and its share of his pitches.
WITH shares AS (
    SELECT pi.name, g.season, p.pitch_class, count(*) AS n,
           count(*) * 1.0 / sum(count(*)) OVER (PARTITION BY pi.name, g.season) AS share
    FROM pitches p
    JOIN games g     ON g.game_pk = p.game_pk
    JOIN pitchers pi ON pi.pitcher_id = p.pitcher_id
    GROUP BY pi.name, g.season, p.pitch_class
)
SELECT name, season, pitch_class, n, round(100 * share, 1) AS pct
FROM (SELECT *, row_number() OVER (PARTITION BY name, season ORDER BY n DESC) AS rk FROM shares) t
WHERE rk = 1
ORDER BY name, season;

-- name: two_strike_pitches_by_batter_hand
-- With two strikes, what does each pitcher throw to lefties vs righties?
SELECT pi.name, p.stand, p.pitch_class, count(*) AS pitches
FROM pitches p
JOIN pitchers pi ON pi.pitcher_id = p.pitcher_id
WHERE p.strikes = 2
GROUP BY pi.name, p.stand, p.pitch_class
ORDER BY pi.name, p.stand, pitches DESC;

-- name: row_counts
-- Sanity check after a load.
SELECT (SELECT count(*) FROM pitchers) AS pitchers,
       (SELECT count(*) FROM games)    AS games,
       (SELECT count(*) FROM pitches)  AS pitches;