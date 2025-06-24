
pub struct RingBuffer<T> {
    elems: Vec<T>,
    capacity: usize,
    read_index: usize,
    write_index: usize
}

#[derive(Debug)]
pub enum RingBufferError {
    Overflow,
    Underflow
}

pub type Result<T> = std::result::Result<T, RingBufferError>;

impl<T: Copy> RingBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        let mut elems = Vec::<T>::with_capacity(capacity+1);
        unsafe { elems.set_len(capacity+1); }
        Self {elems, capacity, read_index: 0, write_index: 0}
    }

    pub fn len(&self) -> usize {
        let a = usize::min(self.read_index, self.write_index);
        let b = usize::max(self.read_index, self.write_index);
        b - a
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn is_empty(&self) -> bool {
        self.read_index == self.write_index
    }

    pub fn is_full(&self) -> bool {
        (self.write_index+1) % (self.capacity+1) == self.read_index
    }

    pub fn clear(&mut self) {
        self.read_index = 0;
        self.write_index = 0;
    }

    pub fn push(&mut self, value: T) -> Result<()> {
        if self.is_full() { return Err(RingBufferError::Overflow); }
        self.elems[self.write_index] = value;
        self.write_index = (self.write_index+1) % (self.capacity+1);
        Ok(())
    }

    pub fn pop(&mut self) -> Result<T> {
        if self.is_empty() { return Err(RingBufferError::Underflow); }
        let ret = self.elems[self.read_index];
        self.read_index = (self.read_index+1) % (self.capacity+1);
        Ok(ret)
    }

    pub fn peek(&self) -> Option<&T> {
        if self.is_empty() { return None; }
        Some(&self.elems[self.read_index])
    }
}

