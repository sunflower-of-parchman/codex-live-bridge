-- Deterministic single suggestion for next best (meter, bpm, mood) gap.
-- BPM range defaults to 60-180.
with normalized_tracks as (
  select
    coalesce(nullif(trim(meter), ''), 'unknown') as meter_raw,
    coalesce(nullif(trim(mood), ''), 'unknown') as mood_raw,
    tempo::int as bpm
  from public.tracks
  where tempo is not null
),
normalized as (
  select
    case
      when meter_raw ~ '^[0-9]+$' then meter_raw || '/4'
      else meter_raw
    end as meter,
    initcap(lower(mood_raw)) as mood,
    bpm
  from normalized_tracks
),
meters as (
  select distinct meter
  from normalized
  where meter <> 'unknown'
),
moods as (
  select distinct mood
  from normalized
  where mood <> 'Unknown'
),
bpms as (
  select generate_series(60, 180) as bpm
),
target as (
  select m.meter, b.bpm, o.mood
  from meters m
  cross join bpms b
  cross join moods o
),
coverage as (
  select meter, bpm, mood, count(*) as track_count
  from normalized
  group by meter, bpm, mood
)
select
  t.meter,
  t.bpm,
  t.mood,
  coalesce(c.track_count, 0) as track_count
from target t
left join coverage c
  on c.meter = t.meter
 and c.bpm = t.bpm
 and c.mood = t.mood
order by track_count asc, t.meter, t.bpm, t.mood
limit 1;
