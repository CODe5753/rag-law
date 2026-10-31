import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import ai_gateway as gw
from gateway_transport import validate_timeout, _request, MAX_RESPONSE_BYTES

class GatewayTests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'AI_GATEWAY_TOKEN': 'synthetic'})
        self.env.start()
    def tearDown(self):
        self.env.stop()
    def test_statuses_explicit(self):
        for status in [401,403,429,502,503,504]:
            with self.subTest(status=status), patch.object(gw, 'post_json', return_value={'status_code':status}):
                with self.assertRaisesRegex(gw.GatewayError, str(status)):
                    gw.generate('answer','synthetic')
    def test_malformed_and_empty(self):
        for data in [{}, {'id':str(uuid4()),'text':''}, {'id':'invalid','text':'answer'}]:
            with patch.object(gw,'post_json',return_value={'status_code':200,'data':data}):
                with self.assertRaises(gw.GatewayError): gw.generate('answer','synthetic')
    def test_grade_schema(self):
        for value in ['{}','{"score":true}','{"score":2}','{"score":"0.9"}','bad','{"score":NaN}']:
            with patch.object(gw,'generate',return_value=value):
                with self.assertRaises(gw.GatewayError): gw.grade('synthetic')
        with patch.object(gw,'generate',return_value='{"score":0.9}'):
            self.assertEqual(gw.grade('synthetic'),.9)
    def test_deadline(self):
        for value in [0,301,float('inf'),float('nan'),True]:
            with self.assertRaises(ValueError): validate_timeout(value)
    def test_bounded_body(self):
        from unittest.mock import MagicMock
        response=MagicMock(); response.status_code=200; response.ok=True
        response.iter_content.return_value=[b'x'*(MAX_RESPONSE_BYTES+1)]
        response.__enter__.return_value=response
        with patch('requests.post',return_value=response):
            self.assertEqual(_request({'url':'http://synthetic','token':'synthetic','payload':{},'timeout':1}), {'error':'invalid_response'})
    def test_business_callers(self):
        from gateway_probe import run
        self.assertEqual(run(True)['mode'],'simulated')
    def test_gateway_failure_not_retried_by_gemini_fallback(self):
        import qdrant_rag as qr
        with patch.dict(os.environ, {'GEMINI_API_KEY':''}), patch.object(qr,'generate',side_effect=gw.GatewayError('AI Gateway HTTP 429')) as call:
            with self.assertRaises(gw.GatewayError): qr.generate_with_gemini('synthetic',[])
            self.assertEqual(call.call_count,1)

if __name__ == '__main__': unittest.main()
