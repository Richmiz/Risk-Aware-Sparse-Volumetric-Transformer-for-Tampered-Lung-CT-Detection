"""F08 metadata-only provenance audit and explicit unfinished research decisions."""

from collections import Counter
import hashlib
import json
from pathlib import Path
from rsvt.data.splits import CLASS_INDEX,audit_partitions,digest_json

ASSIGNMENT_FIELDS=('case_key','source_group_candidate','split','label_candidate','source_file_set_sha256','normalized_sha256')


def audit_f03_metadata(manifest_path=None):
    if manifest_path is None:return {'provided':False,'metadata_only':True,'research_ready':False,'reason':'F03 metadata not supplied'}
    path=Path(manifest_path)
    if path.is_symlink():raise ValueError("F03 manifest symlink rejected")
    raw=path.read_bytes();document=json.loads(raw)
    if document.get('schema_version')!='f03_candidate_v1' or document.get('class_index')!=CLASS_INDEX:
        raise ValueError("F03 schema/class mapping mismatch")
    rows=document['records']
    if not rows or len({r['case_key'] for r in rows})!=len(rows):raise ValueError("Missing/duplicate F03 cases")
    assignments=[{key:r[key] for key in ASSIGNMENT_FIELDS} for r in sorted(rows,key=lambda r:r['case_key'])]
    if digest_json(assignments)!=document.get('assignments_sha256'):raise ValueError("F03 assignment checksum mismatch")
    flags=('classification_eligible','identity_reconciled','label_verified')
    if any(not isinstance(row.get(key),bool) for row in rows for key in flags):raise ValueError("F03 QA flags must be booleans")
    audit=audit_partitions(rows)
    if not audit['ok']:raise ValueError("F03 partition leakage")
    verified=all(row[k] for row in rows for k in flags)
    ready=document.get('training_ready') is True and document.get('status')=='reviewed' and verified and all(r.get('partition_status')=='reviewed' for r in rows)
    return {'provided':True,'metadata_only':True,'volume_files_read':0,'manifest_sha256':hashlib.sha256(raw).hexdigest(),
        'assignments_sha256':document['assignments_sha256'],'cases':len(rows),'distribution':audit['distribution'],
        'qa_true_counts':{key:sum(r[key] for r in rows) for key in flags},'research_ready':ready,
        'generator_candidates':sorted({r['generator_candidate'] for r in rows if r['generator_candidate']}),
        'independent_patients_verified':False,'status':document.get('status','draft')}


def research_readiness(protocol,metadata,catalog):
    pending=[key for key,value in protocol.items() if value=='PENDING']
    reasons=[f'protocol.{key}' for key in pending]
    if protocol.get('status')!='locked':reasons.append('research_protocol_not_locked')
    if protocol.get('selection_partition')!='source_validation_only' or protocol.get('test_used_for_tuning') is not False:
        raise ValueError("Source-only selection and no test tuning are required")
    if not metadata.get('research_ready'):reasons.append('F03_label_identity_and_partition_QA_unfinished')
    if not metadata.get('independent_patients_verified'):reasons.append('canonical_patient_independence_unverified')
    if len(metadata.get('generator_candidates',[]))<2:reasons.append('actual_multi_generator_training_data_missing')
    unsupported=[key for key,value in catalog.items() if not value['entry']['synthetic_backend_supported']]
    reasons.extend(f'backend_or_paired_executor_unavailable:{key}' for key in unsupported)
    # This release deliberately has no medical/test dispatcher, even if a user
    # manually changes draft flags. Future backends require a separate protocol review.
    reasons.append('medical_and_final_test_dispatch_not_implemented')
    return {'research_ready':False,'confirmatory_lock_created':False,'final_test_access_allowed':False,
        'pending_decisions':pending,'blocking_reasons':reasons,'engineering_draft_snapshot_allowed':True}
