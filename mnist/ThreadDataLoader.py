import threading
import torch
import queue

class ThreadDataLoader(torch.utils.data.DataLoader):
    def __iter__(self):
        def load(load_iter, num_batches, q, error_q):
            try:
                # Receive error signal from main thread if an error occurs in the main thread
                if error_q.qsize() > 0:
                    if error_q.get() == 'Error': return
                # Iterate number of batches and insert batches into queue
                for _ in range(num_batches):
                    q.put(next(load_iter))
            # Send error signal to main thread if an error occurs
            except:
                error_q.put("Error")
                raise

        self._current_idx = 0
        self.q = queue.Queue(maxsize=16)
        self.error_q = queue.Queue(maxsize=1)
        self.thread = threading.Thread(target=load, args=(self._get_iterator(), len(self),
                                                          self.q, self.error_q))
        self.thread.start()

        return self

    def __next__(self):
        try:
            # Receive error signal from loader thread if an error occurs in the loader thread
            if self.error_q.qsize() > 0:
                if self.error_q.get() == 'Error': return
            # Iterate number of batches and receive batches from queue
            if self._current_idx < len(self):
                result = self.q.get()
                self._current_idx += 1
                return result
            else:
                raise StopIteration
        except:
            self.error_q.put("Error")
            raise