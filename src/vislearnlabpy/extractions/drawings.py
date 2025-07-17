import pymongo as pm
from pydub import AudioSegment
from io import StringIO
from vislearnlabpy.embeddings.stimuli_loader import ImageExtractor
import io
import base64
from PIL import Image
import os
import numpy as np
import pandas as pd
import datetime
import time
from dataclasses import dataclass
import os
import csv
from pathlib import Path
from tqdm import tqdm

@dataclass
class DrawingTrial:
    session_id: str
    trial_num: int
    category: str
    participantID: str
    filename: str
    submit_time: float
    submit_date: str
    submit_date_readable: str
    start_time: float
    trial_duration: float
    num_strokes: int
    draw_duration: float
    mean_intensity: float
    bounding_box: tuple

@dataclass
class KnowledgeTrial:
    session_id: str
    trial_num: int
    category: str
    participantID: str
    filename: str
    submit_time: float
    submit_date: str
    submit_date_readable: str
    start_time: float
    trial_duration: float

@dataclass
class StrokeData:
    session_id: str
    trial_num: int
    category: str
    stroke_num: int
    filename: str

class DrawingsExtractor():
    @staticmethod
    def get_default_transformation():
        return ImageExtractor.get_transformations(apply_center_crop=False, apply_content_crop=True, use_thumbnail=True)

    @staticmethod
    def save_transformed(imgData, fname, transform=None):
        if transform is None:
            transform = DrawingsExtractor.get_default_transformation()
        
        img_bytes = base64.b64decode(imgData)
        img = Image.open(io.BytesIO(img_bytes))
        img = transform(img)
        img.save(fname)

    @staticmethod
    def load_image_data(imgData, imsize):
        fname = os.path.join('sketch.png')
        with open(fname, "wb") as fh:
            fh.write(base64.b64decode(imgData))
        im = Image.open(fname).resize((imsize, imsize))
        _im = np.array(im)
        return _im
    
    @staticmethod
    def get_mean_intensity(img, imsize):
        thresh = 250
        numpix = imsize**2
        mean_intensity = len(np.where(img[:,:,3].flatten() > thresh)[0]) / numpix
        return mean_intensity
    
    @staticmethod
    def get_bounding_box(img):
        rows = np.any(img, axis=1)
        cols = np.any(img, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        bounding_box = tuple((rmin, rmax, cmin, cmax))
        return bounding_box


class AudioExtractor():
    @staticmethod
    def save_mp3_if_long_enough(audioData, filename="knowledge.mp3", min_duration_sec=1):
        try:
            # If it's a data URL, remove the prefix
            if audioData.startswith("data:"):
                audioData = audioData.split(",")[1]
            
            # Decode base64
            audio_bytes = base64.b64decode(audioData)
            
            # Check if we have any data
            if len(audio_bytes) == 0:
                print("Error: No audio data after base64 decode")
                return False
            
            # Try to detect the actual format first
            try:
                # Try MP3 first
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
            except:
                # If MP3 fails, try other common formats
                for fmt in ["wav", "m4a", "webm", "ogg"]:
                    try:
                        audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
                        break
                    except:
                        continue
                else:
                    print("Error: Could not decode audio in any supported format")
                    return False
            
            # Get duration in seconds
            duration_sec = len(audio) / 1000.0
            
            # Check if it's long enough
            if duration_sec >= min_duration_sec:
                # Convert to MP3 if it wasn't already
                if 'fmt' in locals():  # If we detected a different format
                    audio.export(filename, format="mp3")
                else:
                    with open(filename, "wb") as f:
                        f.write(audio_bytes)
                return True
            else:
                print(f"Audio too short: {duration_sec:.2f} sec (not saved)")
                return False
                
        except Exception as e:
            print(f"Error processing audio: {e}")
            return False


class MongoExtractor():
    def __init__(self, conn_str, database_name, collection_name, output_dir="mongo_output", date=None):
        conn = pm.MongoClient(conn_str)
        self.database = conn[database_name]
        self.collection = self.database[collection_name]
        self.output_dir = output_dir
        if date is not None:
            try:
                self.date = datetime.strptime(date,  "%Y%m%d").date()
            except ValueError:
                raise ValueError("Invalid date format. Please use YYYYMMDD.")
        else:
            self.date = None

    def _is_cdm_run_v3(self):
        """Check if collection is cdm_run_v3"""
        return self.collection.name == 'cdm_run_v3'

    def _get_timing_field(self, start_or_end='start'):
        """Get the appropriate timing field based on collection version"""
        if self._is_cdm_run_v3():
            return 'time'
        else:
            return f'{start_or_end}TrialTime'

    def _get_stroke_timing_field(self, start_or_end='start'):
        """Get the appropriate stroke timing field based on collection version"""
        if self._is_cdm_run_v3():
            return 'time'
        else:
            return f'{start_or_end}StrokeTime'

    def _check_interference(self, session_id):
        """Check for interference in survey data"""
        if self._is_cdm_run_v3():
            return False
        
        survey_session = list(self.collection.find({'$and': [{'dataType':'survey'},{'sessionId':session_id}]}))
        if len(survey_session) > 0:
            return (survey_session[0]['other_drew'] == True | survey_session[0]['parent_drew'] == True)
        return False

    def _create_category_dir(self, base_dir, category_name):
        """Create category directory if it doesn't exist"""
        category_dir = os.path.join(base_dir, category_name)
        if not os.path.exists(category_dir):
            os.makedirs(category_dir)
        return category_dir

    def _should_skip_existing_file(self, filepath, skip_count):
        """Check if file exists and handle skip logic"""
        if os.path.isfile(filepath):
            skip_count += 1
            if np.mod(skip_count, 100) == 0:
                print(f'Skipped {skip_count} files')
            return True, skip_count
        return False, skip_count

    def _get_stroke_records(self, session_id, trial_num):
        """Get stroke records for a given session and trial"""
        timing_field = self._get_timing_field('start')
        return list(self.collection.find({'$and': [
            {'sessionId': session_id}, 
            {'dataType': 'stroke'},
            {'trialNum': trial_num}
        ]}).sort(timing_field))

    def _extract_timing_info(self, record):
        """Extract timing information based on collection version"""
        timing_info = {}
        
        if self._is_cdm_run_v3():
            timing_info['start_time'] = 'NaN'
            timing_info['submit_time'] = record['time']
            timing_info['trial_duration'] = 'NaN'
            timing_info['readable_date'] = datetime.datetime.fromtimestamp(record['time']/1000.0).strftime('%Y-%m-%d %H:%M:%S.%f')
        else:
            timing_info['start_time'] = record['startTrialTime']
            timing_info['submit_time'] = record['endTrialTime']
            timing_info['trial_duration'] = (record['endTrialTime'] - record['startTrialTime'])/1000.00
            timing_info['readable_date'] = datetime.datetime.fromtimestamp(record['endTrialTime']/1000.0).strftime('%Y-%m-%d %H:%M:%S.%f')
        
        return timing_info

    def _calculate_draw_duration(self, stroke_records):
        """Calculate draw duration from stroke records"""
        if len(stroke_records) == 0:
            return 'NA', 'NA'
        
        # Get timing data from strokes
        svg_end_times = []
        svg_start_times = []
        
        for strec in stroke_records:
            if self._is_cdm_run_v3():
                svg_end_times.append(strec['time'])
            else:
                svg_end_times.append(strec['endStrokeTime'])
                svg_start_times.append(strec['startStrokeTime'])
        
        # Calculate durations
        if self._is_cdm_run_v3():
            draw_duration = (svg_end_times[-1] - svg_end_times[0])/1000  # in seconds
        else:
            draw_duration = (svg_end_times[-1] - svg_start_times[0])/1000  # in seconds
        
        return draw_duration

    def _process_image_data(self, img_data, imsize=224):
        """Process image data to get intensity and bounding box"""
        this_image = DrawingsExtractor.load_image_data(img_data, imsize)
        this_intensity = DrawingsExtractor.get_mean_intensity(this_image, imsize)
        
        if this_intensity > 0:
            this_bounding_box = DrawingsExtractor.get_bounding_box(this_image)
        else:
            this_bounding_box = tuple((0, 0, 0, 0))
        
        return this_intensity, this_bounding_box

    def _output_filename(self, filename_prefix, run_name):
        return os.path.join(self.output_dir, f"{filename_prefix}_final_{run_name}.csv")
    
    def _save_dataframe(self, data_dict, filename_prefix, run_name):
        """Save or append rows to <prefix>_final_<run>.csv in output_dir."""
        df = pd.DataFrame(data_dict)
        output_file = self._output_filename(filename_prefix, run_name)

        file_exists = os.path.isfile(output_file)

        # If the file already exists: mode="a" (append) and skip the header.
        # Otherwise: mode="w" (write) and include the header.
        df.to_csv(
            output_file,
            mode="a" if file_exists else "w",
            header=not file_exists,
            index=False,
        )

    def _render_unprocessed_sessions(self, filename_prefix, sessions_to_render):
        output_file = self._output_filename(filename_prefix, self.collection.name)
        if os.path.exists(output_file):
            existing_sessions = pd.read_csv(output_file)
            sessions_to_render = [s for s in sessions_to_render if s not in existing_sessions['session_id'].values]
        return sessions_to_render
    
    def _add_date_query(self, query):
        if self.date is not None:
            # Convert self.date to a timestamp range (start of day to end of day)
            start_of_day = datetime.datetime.combine(self.date, datetime.time.min)
            end_of_day = datetime.datetime.combine(self.date, datetime.time.max)
            query[self._get_timing_field('start')] = {
                '$gte': start_of_day.timestamp() * 1000,
                '$lte': end_of_day.timestamp() * 1000
            }
        return query

    def extract_images(self, image_dir=None, imsize=224, transform_file=False):
        if image_dir is None:
            image_dir = os.path.join(self.output_dir, 'sketches_full_dataset')
        # Initialize tracking variables
        skip_count = 0
        write_count = 0
        interference_count = 0
        
        # Initialize data storage
        trials = []
        query = {'dataType': 'finalImage'}
        # Get all sessions
        sessions_to_render = list(self.collection.find(self._add_date_query(query)).distinct('sessionId'))
        sessions_to_render = self._render_unprocessed_sessions('AllDescriptives_images', sessions_to_render)

        time_start = time.time()

        for session_id in tqdm(sessions_to_render, desc="Drawing sessions processed"):
            # Get image records
            timing_field = self._get_timing_field('start')
            image_recs = list(self.collection.find({'$and': [
                {'sessionId': session_id}, 
                {'dataType': 'finalImage'}
            ]}).sort(timing_field))
            
            # Check for interference
            interference = self._check_interference(session_id)
            if interference:
                interference_count += 1
                if np.mod(interference_count, 10) == 0:
                    print(f'excluded {interference_count} kids for reported interference...')
                continue

            # Process if enough trials and no interference
            if len(image_recs) > 3:
                for imrec in image_recs:
                    if 'category' not in imrec or imrec['category'] is None:
                        continue
                    category_name = "_".join(imrec['category'].split())
                    category_dir = self._create_category_dir(image_dir, category_name)
                    
                    # Create filenames
                    base_filename = f"{category_name}_sketch_{imrec['participantID']}_{imrec['sessionId']}"
                    fname = os.path.join(category_dir, f"{base_filename}.png")
                    
                    # Check if file exists
                    should_skip, skip_count = self._should_skip_existing_file(fname, skip_count)
                    if should_skip:
                        continue
                    # Get stroke records
                    stroke_recs = self._get_stroke_records(session_id, imrec['trialNum'])
                    if len(stroke_recs) > 0:
                        # Extract timing info
                        timing_info = self._extract_timing_info(imrec)
                        
                        # Calculate draw duration
                        draw_duration = self._calculate_draw_duration(stroke_recs)
                        
                        # Process image data
                        intensity, bounding_box = self._process_image_data(imrec['imgData'], imsize)
                        
                        # Store data
                        trials.append(
                            DrawingTrial(
                                imrec["sessionId"], imrec["trialNum"], category_name, imrec["participantID"],
                                fname, timing_info["submit_time"], imrec["date"], timing_info["readable_date"],
                                timing_info["start_time"], timing_info["trial_duration"], len(stroke_recs),
                                draw_duration, intensity, bounding_box
                            )
                        )
                        
                        # Save image files
                        if transform_file:
                            DrawingsExtractor.save_transformed(imrec['imgData'], fname)
                        else:
                            with open(fname, "wb") as fh:
                                fh.write(base64.b64decode(imrec['imgData']))
                        write_count += 1
                        if np.mod(write_count, 100) == 0:
                            time_spent = (time.time() - time_start) / 60
                            print(f'Wrote {write_count} images in {time_spent:.1f} minutes')

            # Save DataFrame after every session write
            self._save_dataframe(trials, 'AllDescriptives_images', self.collection.name)
            trials = []
        print(f"Finished processing {write_count} image files")

    def extract_audio(self, audio_dir=None, min_length=1):
        if audio_dir is None:
            audio_dir = os.path.join(self.output_dir, 'audio_full_dataset')
        # Initialize tracking variables
        skip_count = 0
        write_count = 0
        # Initialize data storage
        trials = []
        # Get all sessions
        query = {'dataType': 'knowledge'}
        sessions_to_render = list(self.collection.find(self._add_date_query(query)).distinct('sessionId'))
        sessions_to_render = self._render_unprocessed_sessions('AllDescriptives_audio', sessions_to_render)
        print(f"Processing {len(sessions_to_render)} sessions for audio extraction") 
        for session_id in tqdm(sessions_to_render, desc="Knowledge trial sessions processed"):
            audio_recs = list(self.collection.find({'$and': [
                {'sessionId': session_id}, 
                {'dataType': 'knowledge'}
            ]}).sort('startTrialTime'))

            for audiorec in audio_recs:
                if 'category' not in audiorec or audiorec['category'] is None:
                    continue
                category_name = "_".join(audiorec['category'].split())
                category_dir = self._create_category_dir(audio_dir, category_name)
                
                # Create filename
                fname = os.path.join(category_dir, f"{category_name}_audio_{audiorec['participantID']}_{audiorec['sessionId']}.mp3")
                
                # Check if file exists
                should_skip, skip_count = self._should_skip_existing_file(fname, skip_count)
                if should_skip:
                    continue
                
                # Try to save audio
                if AudioExtractor.save_mp3_if_long_enough(audiorec['audioData'], filename=fname, min_duration_sec=min_length):
                    write_count += 1
                    
                    # Extract timing info
                    timing_info = self._extract_timing_info(audiorec)
                    
                    # Store data
                    trials.append(KnowledgeTrial(audiorec['sessionId'], audiorec['trialNum'], category_name, audiorec['participantID'], fname, timing_info['submit_time'], audiorec['date'],
                                                 timing_info['readable_date'], timing_info['start_time'], timing_info['trial_duration']))
                else:
                    skip_count += 1
                    trials.append(KnowledgeTrial(audiorec['sessionId'], audiorec['trialNum'], category_name, audiorec['participantID'], 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'))

            # Save DataFrame after every session write
            self._save_dataframe(trials, 'AllDescriptives_audio', self.collection.name)
            trials = []
        print(f"Finished processing {write_count} audio files")

    def extract_strokes(self, session_id, trial_num, save_dir, save_type="csv"):
        """
        Pull stroke documents for *one* trial and save each stroke as either
        a PNG image or a CSV of the SVG path data.
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        # qeury
        sort_key = "time" if self._is_cdm_run_v3() else "startTrialTime"
        stroke_recs = list(
            self.collection
                .find({"sessionId": session_id,
                    "dataType": "stroke",
                    "trialNum": trial_num})
                .sort(sort_key)
        )
        # iterate through strokes
        for i, strec in enumerate(stroke_recs, start=1):
            svg_text = strec["svg"]
            if save_type == "png":
                import cairosvg        
                # e.g. "ABC123_trial7_0001.png"
                out_path = save_dir / f"{session_id}_trial{trial_num}_{i:04d}.png"
                cairosvg.svg2png(bytestring=svg_text.encode(), write_to=str(out_path))

            elif save_type == "csv":
                # one CSV per stroke (change i→None to keep appending to one file)
                out_path = save_dir / f"{session_id}_trial{trial_num}.csv"
                file_exists = out_path.exists()
                mode = "a" if file_exists else "w"
                with open(out_path, mode, newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists:                       # write header once
                        writer.writerow(["session_id", "trial_num", "stroke_num", "svg"])
                    writer.writerow([session_id, trial_num, i, svg_text])
            else:
                raise ValueError("save_type must be 'png' or 'csv'")
