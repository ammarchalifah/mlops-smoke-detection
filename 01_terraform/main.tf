terraform {
  required_version = ">= 1.0"
  backend "local" {}  # Can change from "local" to "gcs" (for google) or "s3" (for aws), if you would like to preserve your tf-state online
  required_providers {
    google = {
      source  = "hashicorp/aws"
    }
  }
}

resource "aws_s3_bucket" "b" {
  bucket = "{AWS_S3_BUCKET}"

  tags = {
    Name = "MLops bucket"
  }
}

resource "aws_s3_bucket_acl" "example" {
  bucket = aws_s3_bucket.b.id
  acl    = "private"
}
