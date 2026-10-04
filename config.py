"""Global constants, action mappings, and theater registries."""

THEATERS = {
    "LADAKH_NORTHERN_AXIS": {
        "center": [34.4800, 77.6500],
        "zoom": 9,
        "nodes": {
            "DEPOT_MAIN": {
                "name": "Rear Staging Base (RSB-01)",
                "lat": 34.1526,
                "lon": 77.5771,
                "role": "Strategic Supply Depot",
            },
            "NODE_TRANSIT": {
                "name": "Intermediate Transit Chokepoint (ILN-North)",
                "lat": 34.3500,
                "lon": 77.6200,
                "role": "Transit Logistics Hub",
            },
            "FOB_ALPHA": {
                "name": "Forward Operating Base (FOB-Alpha)",
                "lat": 34.6000,
                "lon": 77.7000,
                "role": "Forward Echelon Base",
            },
            "POST_SIERRA": {
                "name": "Forward Terminal Post (Sierra)",
                "lat": 34.8500,
                "lon": 77.7800,
                "role": "Frontline Sector Node",
            },
        },
        "primary_route": [
            [34.1526, 77.5771],
            [34.3500, 77.6200],
            [34.6000, 77.7000],
            [34.8500, 77.7800],
        ],
        "alternate_route": [
            [34.1526, 77.5771],
            [34.2800, 77.4500],
            [34.5200, 77.5300],
            [34.8500, 77.7800],
        ],
    },
    "EASTERN_SECTOR_TAWANG": {
        "center": [27.5861, 91.8594],
        "zoom": 9,
        "nodes": {
            "DEPOT_MAIN": {
                "name": "Tezpur Forward Logistics Base",
                "lat": 26.6528,
                "lon": 92.7926,
                "role": "Strategic Supply Depot",
            },
            "NODE_TRANSIT": {
                "name": "Sela Pass Transit Chokepoint",
                "lat": 27.5050,
                "lon": 92.1030,
                "role": "Mountain Pass Transit",
            },
            "FOB_ALPHA": {
                "name": "Tawang Forward Operating Base",
                "lat": 27.5861,
                "lon": 91.8594,
                "role": "Forward Sector Depot",
            },
            "POST_SIERRA": {
                "name": "Bum La Forward Defensive Post",
                "lat": 27.7275,
                "lon": 91.8900,
                "role": "Tactical Frontier Post",
            },
        },
        "primary_route": [
            [26.6528, 92.7926],
            [27.5050, 92.1030],
            [27.5861, 91.8594],
            [27.7275, 91.8900],
        ],
        "alternate_route": [
            [26.6528, 92.7926],
            [27.3500, 92.4000],
            [27.6200, 91.9500],
            [27.7275, 91.8900],
        ],
    },
}

ACTIONS = [
    "NORMAL SUPPLY",
    "FORWARD REPLENISHMENT",
    "PRIORITY REPLENISHMENT",
    "ALTERNATE ROUTE",
    "RESERVE CAPACITY",
]

ACTION_IDS = {action: idx for idx, action in enumerate(ACTIONS)}

STATE_BINS = 4
ALPHA = 0.12
GAMMA = 0.93
