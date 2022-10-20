class Environment:
    def __init__(self, trackfile):
        self.file = trackfile
        self.lines = self.convert_trackfile()
        self.track = self.build_track()
        self.rmap = { "0" : -100, "1" : -1, "f" : 1000, "s" : 0 }
        self.start = self.get_start()
        
    def get_track(self):
        return self.track
        
    def convert_trackfile(self):
        lines = []
        with open(self.file) as f:
            lines = f.readlines()
        return lines

    def print_track(self):
        for l in self.lines:
            print(l, end="")

    def build_track(self):
        rows = len(self.lines)
        cols = len(self.lines[0])
        track_to_numpy = []
        for row in range(len(self.lines)):
            r = []
            for col in range(len(self.lines[row])):
                if self.lines[row][col] != "\n":
                    r.append(self.lines[row][col]) 
            track_to_numpy.append(r)
        return np.asarray(track_to_numpy)
    
    def get_start(self):
        ixs = np.argwhere(self.track=='s').tolist()
        i = np.random.choice(len(ixs), 1)
        s = ixs[i[0]]
        s.extend([0,0])
        self.start = s
        return self.start
            
    def get_reward(self, x, y):
        mpos = self.track[x][y]
        return self.rmap[mpos]
    
    def get_bounds(self):
        return self.track.shape
    
    def check_out_of_bounds(self, x, y):
        return (x<0) or (y<0) or (x>self.get_bounds()[0]-1) or (y>self.get_bounds()[1]-1) or (self.track[x][y]=='0')
    
    def check_not_finished(self, x, y):
        finished = self.track[x][y] == 'f'
        return not finished
    
    def _DEBUGGER_get_end_indices(self):
        return np.argwhere(self.track=='f')
    
    def print_finish(self, x, y):
        for row in range(len(self.track)):
            s = ""
            for col in range(len(self.track[row])):
                if (row==x and col==y):
                    s += "A"   
                else:
                    s += str(self.track[row][col])
            print(s)
        print()
        
