# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Note: this script could only run after setting nn-Meter builder up and creating workspace.
from nn_meter.builder.backend_meta.fusion_rule_tester import generate_testcases, detect_fusion_rule
from nn_meter.builder.backends import connect_backend
import sys

from nn_meter.builder import builder_config, profile_models
builder_config.init(sys.argv[1])  # initialize builder config with workspace

# generate testcases
origin_testcases = generate_testcases()

# connect to backend
backend = connect_backend(backend_name='debug_backend')

# run testcases and collect profiling results
profiled_results = profile_models(backend, origin_testcases, mode='ruletest')

# determine fusion rules from profiling results
detected_results = detect_fusion_rule(profiled_results)
