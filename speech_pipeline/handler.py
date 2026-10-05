# speech_pipeline/handler.py
"""One Lambda function, two S3 triggers.

  audio/<name>.mp3 uploaded  -> start an Amazon Transcribe job -> transcripts/<name>.json
  text/<name>.txt uploaded   -> Amazon Polly speaks it         -> speech/<name>.mp3
"""
import json
import os
import re
import urllib.parse
import uuid

import boto3

s3 = boto3.client("s3")
transcribe = boto3.client("transcribe")
polly = boto3.client("polly")

VOICE_ID = os.environ.get("POLLY_VOICE_ID", "Joanna")
LANGUAGE = os.environ.get("TRANSCRIBE_LANGUAGE", "en-US")


def handle_audio(bucket: str, key: str) -> dict:
    stem = os.path.splitext(os.path.basename(key))[0]
    job_name = re.sub(r"[^0-9a-zA-Z._-]", "-", f"{stem}-{uuid.uuid4().hex[:8]}")
    transcribe.start_transcription_job(
        TranscriptionJobName=job_name,
        Media={"MediaFileUri": f"s3://{bucket}/{key}"},
        MediaFormat=os.path.splitext(key)[1].lstrip(".").lower() or "mp3",
        LanguageCode=LANGUAGE,
        OutputBucketName=bucket,
        OutputKey=f"transcripts/{stem}.json",
    )
    return {"started": job_name}


def handle_text(bucket: str, key: str) -> dict:
    text = s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8")[:3000]
    audio = polly.synthesize_speech(Text=text, OutputFormat="mp3", VoiceId=VOICE_ID, Engine="neural")
    stem = os.path.splitext(os.path.basename(key))[0]
    out_key = f"speech/{stem}.mp3"
    s3.put_object(Bucket=bucket, Key=out_key, Body=audio["AudioStream"].read(),
                  ContentType="audio/mpeg")
    return {"wrote": out_key}


def lambda_handler(event, context):
    results = []
    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])
        if key.startswith("audio/"):
            results.append(handle_audio(bucket, key))
        elif key.startswith("text/"):
            results.append(handle_text(bucket, key))
    print(json.dumps(results))
    return {"results": results}
