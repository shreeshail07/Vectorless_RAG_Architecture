from __future__ import annotations

import csv
import random
from pathlib import Path

from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak

DATA_DIR = Path(__file__).parent / "data"
random.seed(42)


STATES = [
  "Texas", "California", "Florida", "New York", "Illinois",
  "Ohio", "Georgia", "Arizona", "North Carolina", "Michigan",
  "Pennsylvania", "Colorado", "Washington", "Nevada", "Tennessee",
]

LOSS_TYPES = [
  "Collision", "Comprehensive", " Theft",  "Fire", "Flood",
  "Vandalism", 
]

STATUSES = ["Open", "Closed", "Under_review", "Denied", "Pending"]

VEHICLE_TYPES = [
  "Sedan", "SUV", "Pickup Truck", "Van", "Motorcycle", "Coupe"
]



