"""Exercise the deployed contract using synthetic values only; do not print credentials."""
import argparse
import hashlib
import hmac
import json
import os
import time
import urllib.request
import urllib.error
import uuid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--production', action='store_true')
    parser.add_argument('--postgres', action='store_true', help='Require an actual PostgreSQL connection')
    parser.add_argument('--review', action='store_true', help='Test admin review on an isolated test service')
    args = parser.parse_args()
    if args.review and args.production:
        parser.error('Review testing requires an isolated test service')
    device = 'smoke-' + str(uuid.uuid4())
    credential = None

    def request(path, data=None, signed=False, admin=False):
        body = json.dumps(data, separators=(',', ':')).encode() if data is not None else b''
        headers = {'Content-Type': 'application/json'}
        if admin:
            headers['Authorization'] = 'Bearer ' + os.environ['AE_TEST_ADMIN_KEY']
        if signed:
            ts = str(time.time())
            headers.update({'X-Machine-ID': device, 'X-Extension-Timestamp': ts,
                'X-Extension-Signature': hmac.new(credential.encode(), ts.encode()+b'.'+body, hashlib.sha256).hexdigest()})
        req = urllib.request.Request(args.url.rstrip('/')+path,
            data=body if data is not None else None, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            return error.code, json.load(error)

    status, health = request('/readiness')
    assert status == 200 and health['llm_ready'] and health['version'] == '2.2.4', health
    status, stats = request('/stats')
    assert status == 200 and stats['model']['is_trained'] and stats['model']['input_mode'] == 'features_v2'
    assert stats['training']['training_samples_count'] > 0
    if args.production or args.postgres:
        assert stats['database']['backend'] == 'postgresql'
    if args.production:
        assert stats['service']['environment'] == 'production'
        assert stats['cache']['redis_status'] == 'connected'
    assert request('/train', {})[0] == 401
    status, registration = request('/extension/register', {'machine_id': device})
    assert status == 200
    credential = registration['client_secret']
    assert request('/extension/register', {'machine_id': device})[0] == 401
    for value, name, expected in [('sk-SyntheticSmokeValue0123456789AbCdEf', 'API_KEY', True),
                                   ('hello-world', 'greeting', False)]:
        status, result = request('/extension/analyze', {
            'secret_value': value, 'context': f'const {name} = "[REDACTED]";', 'variable_name': name}, True)
        assert status == 200 and result['is_likely_secret'] == expected, {'name': name, 'result': result}
    assert request('/extension/feedback', {'samples': [{
        'id': 'invalid-raw', 'secret_value': 'must-be-rejected', 'features': [0.5]*35,
        'user_action': 'confirmed_secret', 'label': 'high'}]}, True)[0] == 422
    # The synthetic observation is quarantined; this does not update live weights.
    status, feedback = request('/extension/feedback', {'samples': [{
        'id': 'synthetic-smoke-observation', 'feature_schema': 2, 'features': [0.5]*35,
        'user_action': 'marked_false_positive', 'label': 'false_positive'}]}, True)
    assert status == 200 and feedback['status'] == 'queued' and feedback['model_updated'] is False
    assert feedback['accepted_sample_ids'] == ['synthetic-smoke-observation']
    if args.review:
        assert stats['service']['environment'] == 'test'
        status, pending = request('/feedback/pending?limit=100', admin=True)
        assert status == 200
        ids = [row['id'] for row in pending['samples']]
        assert ids
        status, reviewed = request('/feedback/review', {'ids': ids, 'approve': True}, admin=True)
        assert status == 200 and reviewed['model_updated'] and reviewed['processed'] == len(ids)
        status, repeated = request('/feedback/review', {'ids': ids, 'approve': True}, admin=True)
        assert status == 200 and repeated['processed'] == 0 and not repeated['model_updated']
        status, updated = request('/stats')
        assert status == 200 and updated['training']['training_samples_count'] > stats['training']['training_samples_count']
    status, blacklist = request('/extension/blacklist', signed=True)
    assert status == 200 and blacklist['hash_version'] == 2
    print(json.dumps({'status': 'passed', 'version': health['version'],
        'trained': stats['model']['is_trained'], 'environment': stats['service']['environment'],
        'redis': stats['cache']['redis_status'], 'registered_device': device,
        'checks': ['readiness', 'trained checkpoint', 'admin authentication', 'ownership proof',
                   'positive and negative classification', 'raw feedback rejection', 'quarantined feedback', 'blacklist v2']}, indent=2))


if __name__ == '__main__':
    main()
