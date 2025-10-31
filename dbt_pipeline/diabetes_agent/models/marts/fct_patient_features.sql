with e as (
    select * from {{ ref('fct_readmission_labels') }}
),
p as (
    select * from {{ ref('stg_patients') }}
),
enc as (
    select * from {{ ref('stg_encounters') }}
)

select
    e.encounter_id,
    e.patient,
    p.gender,
    p.race,
    year(current_timestamp) - year(p.dob) as age,
    e.days_to_next,
    e.readmitted_within_30d,
    -- Add encounter details for NLP
    enc.encounterclass,
    enc.encounter_description,
    enc.reasoncode,
    enc.reasondescription,
    enc.base_cost,
    enc.total_cost
from e
left join p on e.patient = p.patient_id
left join enc on e.encounter_id = enc.encounter_id

