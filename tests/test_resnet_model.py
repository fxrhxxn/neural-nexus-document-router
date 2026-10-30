import unittest

import torch

from src.models.resnet_model import build_resnet18, get_device


class TestResNetModel(unittest.TestCase):

    def test_device_is_valid(self):
        device = get_device()

        self.assertIn(
            device.type,
            ["cuda", "mps", "cpu"]
        )

    def test_five_class_output_shape(self):
        model = build_resnet18(
            num_classes=5,
            pretrained=False
        )

        dummy_images = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            outputs = model(dummy_images)

        self.assertEqual(outputs.shape, (2, 5))

    def test_nine_class_output_shape(self):
        model = build_resnet18(
            num_classes=9,
            pretrained=False
        )

        dummy_images = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            outputs = model(dummy_images)

        self.assertEqual(outputs.shape, (2, 9))

    def test_backbone_freezing(self):
        model = build_resnet18(
            num_classes=5,
            pretrained=False,
            freeze_backbone=True
        )

        backbone_parameters = [
            parameter.requires_grad
            for name, parameter in model.named_parameters()
            if not name.startswith("fc.")
        ]

        classifier_parameters = [
            parameter.requires_grad
            for name, parameter in model.named_parameters()
            if name.startswith("fc.")
        ]

        self.assertTrue(
            all(not value for value in backbone_parameters)
        )

        self.assertTrue(
            all(classifier_parameters)
        )


if __name__ == "__main__":
    unittest.main()
