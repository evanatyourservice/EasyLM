#!/bin/bash

gcloud auth application-default login
mkdir "$HOME/bucket"
gcsfuse uscentral1stuff "$HOME/bucket"