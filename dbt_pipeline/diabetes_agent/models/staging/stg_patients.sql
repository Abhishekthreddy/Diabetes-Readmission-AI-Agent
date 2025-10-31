with raw as (
    select * 
    from read_parquet('../../data/processed/synthea/patients.parquet')
)
select
    id as patient_id,
    birthdate,
    gender,
    race,
    address,
    city,
    state,
    cast(birthdate as timestamp) as dob
from raw

