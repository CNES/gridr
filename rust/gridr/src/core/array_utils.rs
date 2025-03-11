#![warn(missing_docs)]
//! Crate doc
    
/// Function array1_replace
/// This methods only use rust standard types
/// This methods replaces values in a 1d array by applying the following rule:
/// - if an element equals to 'val_cond' then the element is set to 'val_true'
/// - otherwise the element is set to 'val_false'
/// This methods has been implemented to respond to a lack of python's numpy
/// methods (where, putmask copyto) that allocate temporary memory.
pub fn array1_replace<T>(
        array: &mut [T], 
        val_cond: T,
        val_true: T,
        val_false: T) -> ()
where
    T: Copy + PartialEq,
{
    for elem in array.iter_mut() {
          *elem = if *elem == val_cond { val_true } else { val_false };
    }
    ()
}

