# tests/test_aws.py
import io
import os

import pytest
from botocore.response import StreamingBody
from botocore.stub import ANY, Stubber

os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "test")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "test")

from app.providers.base import CallRequest, Message  # noqa: E402
from app.providers.bedrock import BedrockProvider  # noqa: E402
from speech_pipeline import handler  # noqa: E402


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_bedrock_converse_shape():
    p = BedrockProvider("us-east-1")
    with Stubber(p.client) as stub:
        stub.add_response(
            "converse",
            {"output": {"message": {"role": "assistant", "content": [{"text": "Paris"}]}},
             "stopReason": "end_turn",
             "usage": {"inputTokens": 12, "outputTokens": 2, "totalTokens": 14},
             "metrics": {"latencyMs": 100}},
            {"modelId": "us.amazon.nova-micro-v1:0",
             "messages": [{"role": "user", "content": [{"text": "Capital of France?"}]}],
             "system": [{"text": "Be brief."}],
             "inferenceConfig": {"maxTokens": 50, "temperature": 0.2, "topP": 1.0}},
        )
        res = await p.chat(CallRequest(
            [Message("system", "Be brief."), Message("user", "Capital of France?")],
            "us.amazon.nova-micro-v1:0", max_tokens=50))
    assert res.text == "Paris" and res.usage.output_tokens == 2


def _event(key):
    return {"Records": [{"s3": {"bucket": {"name": "b"}, "object": {"key": key}}}]}


def test_audio_upload_starts_transcribe_job():
    with Stubber(handler.transcribe) as stub:
        stub.add_response("start_transcription_job", {"TranscriptionJob": {}},
                          {"TranscriptionJobName": ANY, "Media": {"MediaFileUri": "s3://b/audio/my call.mp3"},
                           "MediaFormat": "mp3", "LanguageCode": "en-US",
                           "OutputBucketName": "b", "OutputKey": "transcripts/my call.json"})
        out = handler.lambda_handler(_event("audio/my+call.mp3"), None)
    assert out["results"][0]["started"].startswith("my-call-")


def test_text_upload_writes_polly_audio():
    body = StreamingBody(io.BytesIO(b"Hello class"), 11)
    with Stubber(handler.s3) as s3s, Stubber(handler.polly) as ps:
        s3s.add_response("get_object", {"Body": body}, {"Bucket": "b", "Key": "text/hello.txt"})
        ps.add_response("synthesize_speech", {"AudioStream": StreamingBody(io.BytesIO(b"ID3"), 3)},
                        {"Text": "Hello class", "OutputFormat": "mp3", "VoiceId": "Joanna", "Engine": "neural"})
        s3s.add_response("put_object", {}, {"Bucket": "b", "Key": "speech/hello.mp3", "Body": b"ID3",
                                            "ContentType": "audio/mpeg"})
        out = handler.lambda_handler(_event("text/hello.txt"), None)
    assert out["results"] == [{"wrote": "speech/hello.mp3"}]


@pytest.mark.anyio
async def test_bedrock_counts_cached_tokens_and_streams():
    p = BedrockProvider("us-east-1")
    with Stubber(p.client) as stub:
        stub.add_response(
            "converse",
            {"output": {"message": {"role": "assistant", "content": [{"text": "ok"}]}},
             "stopReason": "end_turn",
             "usage": {"inputTokens": 10, "outputTokens": 2, "totalTokens": 1212,
                       "cacheReadInputTokens": 1200},
             "metrics": {"latencyMs": 1}},
        )
        res = await p.chat(CallRequest([Message("user", "hi")], "m"))
    assert res.usage.input_tokens == 1210 and res.usage.cached_input_tokens == 1200

    class FakeClient:
        def converse_stream(self, **kwargs):
            return {"stream": [{"messageStart": {}},
                               {"contentBlockDelta": {"delta": {"text": "Hel"}}},
                               {"contentBlockDelta": {"delta": {"text": "lo"}}},
                               {"messageStop": {}}]}
    p.client = FakeClient()
    pieces = [x async for x in p.stream(CallRequest([Message("user", "hi")], "m"))]
    assert "".join(pieces) == "Hello"
