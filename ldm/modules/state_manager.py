class DiffusionStateManager:
    def __init__(self) -> None:
        #self.inputs = set()
        #self.destination = None
        self.input_mapping = {}
        self.stored_values = {}

    def set_destination(self, input_name, destination):
        self.input_mapping[input_name] = destination

    def send(self, source_id, input_name, value=None):
        if input_name in self.input_mapping:
            self.input_mapping[input_name](source_id, value)

    def store(self, for_source_id, input_name, value=None):
        if for_source_id not in self.stored_values:
            self.stored_values[for_source_id] = {}
        self.stored_values[for_source_id][input_name] = value

    def recv(self, source_id, input_name):
        if source_id in self.stored_values and input_name in self.stored_values[source_id]:
            return self.stored_values[source_id][input_name]
        return None

    def reset(self):
        del self.input_mapping
        del self.stored_values

        self.input_mapping = {}
        self.stored_values = {}

    @staticmethod
    def use_or_create(other):
        if other is not None:
            return other
        else:
            raise Exception('other mgr is None')
            #return DiffusionStateManager()
