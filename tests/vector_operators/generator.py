#!/usr/bin/env python3
# ************************************************************************
#
#   SYCL Conformance Test Suite
#
#
#   Copyright (c) 2023 The Khronos Group Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
#
# ************************************************************************

import sys
import argparse
from string import Template

sys.path.append('../common/')
from common_python_vec import (Data, ReverseData, remove_namespaces_whitespaces, get_types)


def generate_operator_tests(type_str, input_file, output_file):
    test_cases = ''


    with open(input_file, 'r') as f:
        template = Template(f.read())

    result = template.substitute(
        category='operators',
        type_name=remove_namespaces_whitespaces(type_str),
        test_cases=test_cases)

    with open(output_file, 'w+') as f:
        f.write(result)


def main():
    argparser = argparse.ArgumentParser(
        description='Generates tests for sycl::vec operators')
    argparser.add_argument(
        'template',
        metavar='<code template path>',
        help='Path to code template')
    argparser.add_argument(
        '-type',
        dest='ty',
        required=True,
        choices=get_types(),
        help='Type to generate the test for')
    argparser.add_argument(
        '-o',
        required=True,
        dest="output",
        metavar='<out file>',
        help='CTS test output')
    args = argparser.parse_args()

    generate_operator_tests(args.ty, args.template, args.output)


if __name__ == '__main__':
    main()
