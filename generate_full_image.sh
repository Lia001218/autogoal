#!/bin/bash
contribs=(sklearn nltk gensim regex keras )
docker build . -t autogoal/autogoal:full_image -f dockerfiles/development/dockerfile --build-arg extras="common $contribs remote" --no-cache
