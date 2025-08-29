# Predictor2 is finalized
The main model is Predictor2 and will be refering only to Predictor2 henceforth

This Python script reads the CanSat telemetry file row-by-row, and for each telemetry snapshot it uses simple physics (parachute terminal velocity + horizontal drift estimated from GPS movement) to predict where the payload will land (latitude & longitude). It also writes per-row predictions to a CSV and prints a final prediction from the last row. 

NOTE THAT THIS SCRIPT IS WRITTEN IN REFERENCE TO THE EXCEL FILE(cansat_telemetry.xlsx), FOR DIFFERENT FORM OF DATA, WE WILL NEED TO CHANGE VARIABLE NAMES.  FOR RUNNING ON YOUR COMPUTER, ADD YOUR OWN PATH AFTER SAVING TO LINE NUMBER 26 of PREDICTOR2.

Flow
Load telemetry (Excel).
For each telemetry row (timestamp, lat, lon, altitude):
 a. Estimate horizontal speed & direction from GPS change.
 b. Estimate vertical speed (vz) from altitude change.
 c. Estimate where the rocket will reach apogee (if still going up).
 d. Compute parachute terminal velocity (using mass, parachute Cd & area, and air density).
 e. Compute time to descend from apogee to ground = distance / terminal velocity.
 f. Compute horizontal drift = horizontal_speed × descent_time.
 g. Convert drift (meters + bearing) into a predicted (lat, lon) landing coordinate.
Save and print results.


Physics Constants

G = 9.81 → Earth gravity (m/s²). Objects accelerate downwards ~9.81 m/s each second (ignoring drag).

R_EARTH = 6,371,000.0 → Earth’s radius (m). Used to convert GPS lat/lon into X–Y distances.

RHO0 = 1.225 → Air density at sea level (kg/m³). Thins with altitude → less drag.

SCALE_HEIGHT = 8500.0 → Scale height (m). At ~8.5 km higher, air density drops to ~37% of sea-level value.


 
Telemetry Columns

TIME_COL = "timestamp" → Time of each reading.

LAT_COL = "lat" → Latitude (north–south).

LON_COL = "long" → Longitude (east–west).

ALT_COL = "altitude" → Altitude above ground/sea.

(Using variables avoids hardcoding column names in the code.)
 

The Output file is given by OUTPUT_CSV = "predictions_per_row.csv" at line 48

FURTHER WORK TO BE DONE


