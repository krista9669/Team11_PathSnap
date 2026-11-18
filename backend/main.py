
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List
from pathlib import Path
import math


# ==========================================
# 1. APPLICATION CONFIGURATION
# ==========================================

app = FastAPI(
    title="PathSnap API",
    description="GPS Trajectory Map Matching and Evaluation System",
    version="2.1.0"
)

# Dataset directory
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "evaluator_test"

# Four required files for every evaluation trip
REQUIRED_EXTENSIONS = [
    "track",
    "arcs",
    "nodes",
    "route"
]


# ==========================================
# 2. REQUEST MODELS
# ==========================================

class GPSPoint(BaseModel):

    latitude: float = Field(ge=-90, le=90)

    longitude: float = Field(ge=-180, le=180)

    point_index: int = Field(ge=1)


class PredictionRequest(BaseModel):

    trip_id: str = Field(min_length=1)


# ==========================================
# 3. DATASET HELPER FUNCTIONS
# ==========================================

def validate_trip_id(trip_id: str):

    if not trip_id.isdigit():

        raise HTTPException(
            status_code=400,
            detail="Invalid trip ID."
        )


def get_trip_files(trip_id: str):

    validate_trip_id(trip_id)

    trip_folder = DATASET_DIR / trip_id

    if not trip_folder.is_dir():

        raise HTTPException(
            status_code=404,
            detail=f"Trip {trip_id} not found."
        )

    files = {

        extension: trip_folder / f"{trip_id}.{extension}"

        for extension in REQUIRED_EXTENSIONS

    }

    missing_files = [

        extension

        for extension, file_path in files.items()

        if not file_path.is_file()

    ]

    if missing_files:

        raise HTTPException(
            status_code=500,
            detail={
                "message": f"Trip {trip_id} has missing files.",
                "missing_files": missing_files
            }
        )

    return files


# ==========================================
# 4. GPS TRACK LOADER
# ==========================================

def load_gps_track(track_path: Path):

    gps_points = []

    with open(track_path, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            values = line.split()

            if not values:
                continue

            # Every observation must have 3 columns
            if len(values) != 3:

                raise HTTPException(
                    status_code=500,
                    detail=f"Invalid GPS format at line {line_number}."
                )

            try:

                longitude = float(values[0])

                latitude = float(values[1])

                point_index = float(values[2])

            except ValueError:

                raise HTTPException(
                    status_code=500,
                    detail=f"Non-numeric GPS value at line {line_number}."
                )

            # Check for NaN and infinity
            if not all(map(math.isfinite, [
                longitude,
                latitude,
                point_index
            ])):

                raise HTTPException(
                    status_code=500,
                    detail=f"Invalid numeric value at line {line_number}."
                )

            # Validate longitude
            if not -180 <= longitude <= 180:

                raise HTTPException(
                    status_code=500,
                    detail=f"Invalid longitude at line {line_number}."
                )

            # Validate latitude
            if not -90 <= latitude <= 90:

                raise HTTPException(
                    status_code=500,
                    detail=f"Invalid latitude at line {line_number}."
                )

            # ==================================
            # FLOATING-POINT PRECISION FIX
            # ==================================

            rounded_index = round(point_index)

            # Allow tiny floating-point differences
            # Example: 452.999 becomes 453
            # Example: 453.999 becomes 454

            if (
                rounded_index < 1
                or abs(point_index - rounded_index) > 0.01
            ):

                raise HTTPException(
                    status_code=500,
                    detail=f"Invalid point index at line {line_number}."
                )

            gps_points.append({

                "latitude": latitude,

                "longitude": longitude,

                "point_index": int(rounded_index)

            })

    # Check empty GPS file
    if not gps_points:

        raise HTTPException(
            status_code=500,
            detail="GPS track file is empty."
        )

    # Sort observations by point index
    gps_points.sort(
        key=lambda point: point["point_index"]
    )

    return gps_points


# ==========================================
# 5. HOME ENDPOINT
# ==========================================

@app.get("/")
def home():

    return {

        "project": "PathSnap",

        "message": "Welcome to PathSnap API",

        "status": "active",

        "version": "2.1.0"

    }


# ==========================================
# 6. HEALTH ENDPOINT
# ==========================================

@app.get("/health")
def health():

    return {

        "status": "healthy",

        "dataset_directory": str(DATASET_DIR),

        "dataset_exists": DATASET_DIR.is_dir()

    }


# ==========================================
# 7. GET ALL EVALUATION TRIPS
# ==========================================

@app.get("/trips")
def get_all_trips():

    if not DATASET_DIR.is_dir():

        raise HTTPException(
            status_code=500,
            detail="Evaluation dataset directory not found."
        )

    trips = []

    # Find all numerical trip folders
    trip_folders = sorted(

        [
            folder

            for folder in DATASET_DIR.iterdir()

            if folder.is_dir() and folder.name.isdigit()

        ],

        key=lambda folder: int(folder.name)

    )

    for trip_folder in trip_folders:

        trip_id = trip_folder.name

        # Check all four required files
        missing_files = [

            extension

            for extension in REQUIRED_EXTENSIONS

            if not (
                trip_folder / f"{trip_id}.{extension}"
            ).is_file()

        ]

        if missing_files:

            trips.append({

                "trip_id": trip_id,

                "number_of_gps_points": None,

                "status": "unavailable",

                "missing_files": missing_files

            })

            continue

        try:

            track_path = (
                trip_folder / f"{trip_id}.track"
            )

            gps_points = load_gps_track(track_path)

            trips.append({

                "trip_id": trip_id,

                "number_of_gps_points": len(gps_points),

                "status": "available",

                "missing_files": []

            })

        except (HTTPException, OSError) as error:

            trips.append({

                "trip_id": trip_id,

                "number_of_gps_points": None,

                "status": "invalid",

                "error": (
                    error.detail
                    if isinstance(error, HTTPException)
                    else str(error)
                )

            })

    # Calculate trip availability
    available_count = sum(

        1

        for trip in trips

        if trip["status"] == "available"

    )

    return {

        "total_trips": len(trips),

        "available_trips": available_count,

        "unavailable_trips": (
            len(trips) - available_count
        ),

        "trips": trips

    }


# ==========================================
# 8. GET SELECTED EVALUATION TRIP
# ==========================================

@app.get("/trips/{trip_id}")
def get_selected_trip(trip_id: str):

    # Validate and locate trip files
    files = get_trip_files(trip_id)

    # Load original GPS observations
    gps_points = load_gps_track(
        files["track"]
    )

    # Starting and ending observations
    start_point = gps_points[0]

    end_point = gps_points[-1]

    return {

        "trip_id": trip_id,

        "number_of_gps_points": len(gps_points),

        "gps_points": gps_points,

        "starting_coordinates": {

            "latitude": start_point["latitude"],

            "longitude": start_point["longitude"]

        },

        "ending_coordinates": {

            "latitude": end_point["latitude"],

            "longitude": end_point["longitude"]

        },

        "reference_route_available": True,

        "road_network_available": True,

        "status": "loaded",

        "message": "Evaluation trip loaded successfully."

    }


# ==========================================
# 9. PREDICTION ENDPOINT
# ==========================================

@app.post("/predict")
def predict(request: PredictionRequest):

    trip_id = request.trip_id

    # Validate selected trip
    files = get_trip_files(trip_id)

    # Load GPS observations
    gps_points = load_gps_track(
        files["track"]
    )

    start_point = gps_points[0]

    end_point = gps_points[-1]

    # ML and HMM prediction are pending integration
    return {

        "trip_id": trip_id,

        "number_of_gps_points": len(gps_points),

        "status": "awaiting_model_integration",

        "predicted_arc_ids": None,

        "starting_coordinates": {

            "latitude": start_point["latitude"],

            "longitude": start_point["longitude"]

        },

        "ending_coordinates": {

            "latitude": end_point["latitude"],

            "longitude": end_point["longitude"]

        },

        "reference_route_available": True,

        "accuracy": None,

        "message": (
            "Selected evaluation trip loaded successfully. "
            "Actual ML and HMM/Viterbi prediction "
            "is pending integration."
        )

    }
