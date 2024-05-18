from argparse import ArgumentParser

import torch


def add_commands(parser: ArgumentParser,
                 ) -> ArgumentParser:
    subparsers = parser.add_subparsers(dest="mode",
                                       help="Execution mode")
    subparsers.add_parser("api")
    subparsers.add_parser("worker")
    return parser


def add_database_args(parser: ArgumentParser,
                      ) -> ArgumentParser:
    group = parser.add_argument_group("Vector database options")
    group.add_argument("--database-url",
                       type=str,
                       default="localhost:6333",
                       help="Complete URL of Qdrant server")
    group.add_argument("--collection",
                       type=str,
                       default="default",
                       help="Qdrant collection name")
    return parser


def add_embedding_args(parser: ArgumentParser,
                       ) -> ArgumentParser:
    group = parser.add_argument_group("Embedding options")
    group.add_argument("--model",
                       type=str,
                       default="openai/clip-vit-base-patch32",
                       help="Model name or path")
    group.add_argument("--device",
                       type=torch.device,
                       default=torch.device("cpu"),
                       help="Device to use for inference")
    return parser


def add_queue_args(parser: ArgumentParser,
                   ) -> ArgumentParser:
    group = parser.add_argument_group("Task queue options")
    group.add_argument("--broker-url",
                       type=str,
                       default="redis://localhost",
                       help="Broker URL for task queue")
    return parser
