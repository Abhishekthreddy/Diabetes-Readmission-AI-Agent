with raw as (
    select * 
    from read_parquet('../../data/processed/synthea/encounters.parquet')
)
select
    "Id" as encounter_id,
    "PATIENT" as patient,
    "ENCOUNTERCLASS" as encounterclass,
    "CODE" as encounter_code,
    "DESCRIPTION" as encounter_description,
    "REASONCODE" as reasoncode,
    "REASONDESCRIPTION" as reasondescription,
    "START" as start,
    "STOP" as stop,
    "BASE_ENCOUNTER_COST" as base_cost,
    "TOTAL_CLAIM_COST" as total_cost,
    cast("START" as timestamp) as start_ts,
    cast("STOP" as timestamp) as stop_ts
from raw

