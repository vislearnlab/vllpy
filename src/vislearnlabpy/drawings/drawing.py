import pandas as pd
class Drawing():
    def __init__(self, session_id, trial_num, category, participantID, filename, submit_time, submit_date, submit_date_readable, start_time, trial_duration, num_strokes, draw_duration, mean_intensity, bounding_box):
        self.session_id = session_id
        self.trial_num = trial_num
        self.category = category
        self.participantID = participantID
        self.filename = filename
        self.submit_time = submit_time
        self.submit_date = submit_date
        self.submit_date_readable = submit_date_readable
        self.start_time = start_time
        self.trial_duration = trial_duration
        self.num_strokes = num_strokes
        self.draw_duration = draw_duration
        self.mean_intensity = mean_intensity
        self.bounding_box = bounding_box

class Drawings():
    def __init__(self, drawings=None):
        if drawings is None:
            self.drawings = []
        else:
            self.drawings = drawings
    
    def from_csv(self, csv_file):
        """
        Load drawings from a CSV file.
        The CSV should have columns matching the Drawing attributes.
        """
        df = pd.read_csv(csv_file)
        self.drawings = [Drawing(**row) for _, row in df.iterrows()]

    def add_drawing(self, drawing):
        if isinstance(drawing, Drawing):
            self.drawings.append(drawing)
        else:
            raise TypeError("Expected a Drawing instance")

    def get_drawings(self):
        return self.drawings

    def __len__(self):
        return len(self.drawings)