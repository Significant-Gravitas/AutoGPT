       subroutine t(fun,a)
       integer a
cf2py  intent(out) a
       external fun
       call fun(a)
       end

       subroutine func(a)
cf2py  intent(in,out) a
       integer a
       a = a + 11
       end

       subroutine func0(a)
cf2py  intent(out) a
       integer a
       a = 11
       end

       subroutine t2(a)
cf2py  intent(callback) fun
       integer a
cf2py  intent(out) a
       external fun
       call fun(a)
       end

       subroutine string_callback(callback, a)
       external callback
       double precision callback
       double precision a
       character*1 r
cf2py  intent(out) a
       r = 'r'
       a = callback(r)
       end

       subroutine string_callback_array(callback, cu, lencu, a)
       external callback
       integer callback
       integer lencu
       character*8 cu(lencu)
       integer a
cf2py  intent(out) a

       a = callback(cu, lencu)
       end

       subroutine hidden_callback(a, r)
       external global_f
cf2py  intent(callback, hide) global_f
       integer a, r, global_f
cf2py  intent(out) r
       r = global_f(a)
       end

       subroutine hidden_callback2(a, r)
       external global_f
       integer a, r, global_f
cf2py  intent(out) r
       r = global_f(a)
       end
