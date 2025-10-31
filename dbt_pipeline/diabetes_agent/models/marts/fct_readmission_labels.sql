with visits as (
    select
        *,
        row_number() over (
            partition by patient
            order by start_ts
        ) as visit_num
    from {{ ref('stg_encounters') }}
),

with_leads as (
    select
        a.*,
        b.start_ts as next_visit_ts,
        datediff('day', a.stop_ts, b.start_ts) as days_to_next
    from visits a
    left join visits b
      on a.patient = b.patient
     and a.visit_num + 1 = b.visit_num
)

select
    encounter_id,
    patient,
    start_ts,
    stop_ts,
    days_to_next,
    case 
        when days_to_next <= 30 then 1
        else 0
    end as readmitted_within_30d
from with_leads

