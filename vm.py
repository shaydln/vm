"""
Simplified VM code which works for some cases.
You need extend/rewrite code to pass all cases.
"""

import builtins
import dis
import types
import typing as tp
import operator


class Frame:
    """
    Frame header in cpython with description
        https://github.com/python/cpython/blob/3.11/Include/frameobject.h

    Text description of frame parameters
        https://docs.python.org/3/library/inspect.html?highlight=frame#types-and-members
    """

    def __init__(self,
                 frame_code: types.CodeType,
                 frame_builtins: dict[str, tp.Any],
                 frame_globals: dict[str, tp.Any],
                 frame_locals: dict[str, tp.Any]) -> None:
        self.code = frame_code
        self.builtins = frame_builtins
        self.globals = frame_globals
        self.locals = frame_locals
        self.data_stack: tp.Any = []
        self.return_value = None
        self.index: dict[int, int] = dict()
        self.counter = 0
        self.last_raised_exception = None
        self.kw_names: tuple[tp.Any, ...] = tuple()

    def top(self) -> tp.Any:
        return self.data_stack[-1]

    def pop(self) -> tp.Any:
        return self.data_stack.pop()

    def push(self, *values: tp.Any) -> None:
        self.data_stack.extend(values)

    def popn(self, n: int) -> tp.Any:
        """
        Pop a number of values from the value stack.
        A list of n values is returned, the deepest value first.
        """
        if n > 0:
            returned = self.data_stack[-n:]
            self.data_stack[-n:] = []
            return returned
        else:
            return []

    def run(self) -> tp.Any:
        self.counter -= 1
        lst = list(dis.get_instructions(self.code))
        for i in range(len(lst)):
            self.index[lst[i].offset] = i
        while self.counter < len(lst) - 1:
            self.counter += 1
            f_name = lst[self.counter].opname.lower()
            if f_name + "op" == "return_value_op":
                break
            elif f_name in "load_global":
                getattr(self, f_name + "_op")(lst[self.counter].arg,
                                              lst[self.counter].argval)
            elif f_name == "is_op":
                getattr(self, f_name)(lst[self.counter].arg)
            elif f_name == "format_value":
                getattr(self, f_name + "_op")(lst[self.counter].arg)
            elif f_name == "binary_op" or f_name == "compare_op":
                getattr(self, f_name)(lst[self.counter].arg)
            else:
                getattr(self, f_name + "_op")(lst[self.counter].argval)
        return self.return_value

    def resume_op(self, arg: int) -> tp.Any:
        pass

    def precall_op(self, arg: int) -> tp.Any:
        pass

    def push_null_op(self, arg: int) -> tp.Any:
        self.push(None)

    def call_op(self, arg: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-CALL
        """
        args = self.popn(arg)
        func = self.pop()
        kwargs: dict[str, tp.Any] = {}
        if self.top() is None:
            self.pop()
            self.push(func(*args, **kwargs))
            return
        obj = func
        func = self.pop()
        self.push(func(obj, *args, **kwargs))

    def load_name_op(self, arg: str) -> None:
        """
        Partial realization

        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_NAME
        """
        if arg in self.locals:
            self.push(self.locals[arg])
        elif arg in self.globals:
            self.push(self.globals[arg])
        elif arg in self.builtins:
            self.push(self.builtins[arg])
        else:
            raise NameError

    def load_global_op(self, arg: int, val: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_GLOBAL
        """
        if (arg & 1) == 1:
            self.push_null_op(0)
        if val in self.globals:
            self.push(self.globals[val])
        elif val in self.builtins:
            self.push(self.builtins[val])
        else:
            raise NameError

    def load_const_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-LOAD_CONST
        """
        self.push(arg)

    def return_value_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-RETURN_VALUE
        """
        self.return_value = self.pop()

    def pop_top_op(self, arg: tp.Any) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-POP_TOP
        """
        self.pop()

    def make_function_op(self, flags: int) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-MAKE_FUNCTION
        """
        code = self.pop()

        def create_function(*args: tp.Any, **kwargs: tp.Any) -> tp.Any:
            parsed_args: dict[str, tp.Any] = {}
            names = code.co_varnames
            arg_count = code.co_argcount
            pos_only_count = code.co_posonlyargcount
            kw_only_count = code.co_kwonlyargcount
            default_values = ()
            if hasattr(args, "args_default"):
                default_values = args.args_default
            bool_varkw: bool = bool(code.co_flags & 8)
            bool_varargs: bool = bool(code.co_flags & 4)
            shift = arg_count + kw_only_count - code.co_nlocals
            if bool_varargs:
                vararg_name = names[shift]
                parsed_args[vararg_name] = ()
                shift += 1

            if bool_varkw:
                varkw_name = names[shift]
                parsed_args[varkw_name] = {}
                shift += 1

            if len(args) > arg_count:
                parsed_args[vararg_name] = args[arg_count:len(args)]

            for key in kwargs:
                if key not in names:
                    parsed_args[varkw_name][key] = kwargs[key]
                if names.index(key) < pos_only_count:
                    parsed_args[varkw_name][key] = kwargs[key]
                parsed_args[key] = kwargs[key]

            for i, arg_val in enumerate(args):
                if (bool_varargs
                        and i >= len(args) - len(parsed_args[vararg_name])):
                    break
                parsed_args[names[i]] = arg_val

            for i in range(len(args), arg_count):
                if names[i] not in kwargs:
                    break
                parsed_args[names[i]] = (
                    default_values)[i - arg_count + len(default_values)]

            f_locals = dict(self.locals)
            f_locals.update(parsed_args)
            return Frame(code, self.builtins, self.globals, f_locals).run()

        self.push(create_function)

    def store_name_op(self, arg: str) -> None:
        """
        Operation description:
            https://docs.python.org/release/3.11.5/library/dis.html#opcode-STORE_NAME
        """
        self.locals[arg] = self.pop()

    def store_global_op(self, arg: str) -> None:
        self.globals[arg] = self.pop()

    def store_fast_op(self, arg: str) -> None:
        self.locals[arg] = self.pop()

    BINARY_OPERATORS = {
        0: operator.add,
        1: operator.and_,
        2: operator.floordiv,
        3: operator.lshift,
        4: operator.matmul,
        5: operator.mul,
        6: operator.mod,
        7: operator.or_,
        8: operator.pow,
        9: operator.rshift,
        10: operator.sub,
        11: operator.truediv,
        12: operator.xor,
        13: lambda x, y: x.__add__(y),
        14: lambda x, y: x.__and__(y),
        15: lambda x, y: x.__floordiv__(y),
        16: lambda x, y: x.__lshift__(y),
        17: lambda x, y: x.__matmul__(y),
        18: lambda x, y: x.__mul__(y),
        19: lambda x, y: x.__mod__(y),
        20: lambda x, y: x.__or__(y),
        21: lambda x, y: x.__pow__(y),
        22: lambda x, y: x.__rshift__(y),
        23: lambda x, y: x.__sub__(y),
        24: lambda x, y: x.__truediv__(y),
        25: lambda x, y: x.__xor__(y)
    }

    COMPARE_OPERATORS = {
        0: operator.lt,
        1: operator.le,
        2: operator.eq,
        3: operator.ne,
        4: operator.gt,
        5: operator.ge
    }

    def binary_op(self, arg: int) -> None:
        left, right = self.popn(2)
        self.push(self.BINARY_OPERATORS[arg](left, right))

    def compare_op(self, arg: int) -> None:
        left, right = self.popn(2)
        self.push(self.COMPARE_OPERATORS[arg](left, right))

    def unpack_sequence_op(self, count: int) -> None:
        items = []
        iterator = iter(self.pop())
        while True:
            try:
                items.append(next(iterator))
            except StopIteration:
                break
        for i in items[::-1]:
            self.push(i)

    def kw_names_op(self, arg: tp.Any) -> None:
        self.kw_names = self.code.co_consts[arg]

    def get_iter_op(self, nothing: int) -> None:
        self.push(self.pop().__iter__())

    def jump_forward_op(self, arg: int) -> None:
        self.counter = self.index[arg] - 1

    def jump_backward_op(self, arg: int) -> None:
        self.counter = self.index[arg] - 1

    def jump_backward_no_interrupt_op(self, arg: int) -> None:
        self.counter = self.index[arg] - 1

    def for_iter_op(self, to: int) -> None:
        try:
            value = self.top().__next__()
            self.push(value)
        except StopIteration:
            self.pop()
            self.jump_forward_op(to)

    def pop_jump_forward_if_true_op(self, to: int) -> None:
        if self.pop():
            self.jump_forward_op(to)

    def pop_jump_forward_if_false_op(self, to: int) -> None:
        if not self.pop():
            self.jump_forward_op(to)

    def pop_jump_backward_if_true_op(self, to: int) -> None:
        if self.pop():
            self.jump_forward_op(to)

    def pop_jump_backward_if_false_op(self, to: int) -> None:
        if not self.pop():
            self.jump_forward_op(to)

    def pop_jump_backward_if_none_op(self, to: int) -> None:
        if self.pop() is None:
            self.jump_backward_op(to)

    def pop_jump_backward_if_not_none_op(self, to: int) -> None:
        if self.pop() is not None:
            self.jump_backward_op(to)

    def jump_if_true_or_pop_op(self, to: int) -> None:
        if self.top():
            self.jump_forward_op(to)
        else:
            self.pop()

    def jump_if_false_or_pop_op(self, to: int) -> None:
        if not self.top():
            self.jump_forward_op(to)
        else:
            self.pop()

    def load_fast_op(self, argval: str) -> None:
        self.load_name_op(argval)

    def pop_jump_forward_if_none_op(self, to: int) -> None:
        if self.pop() is None:
            self.jump_forward_op(to)

    def pop_jump_forward_if_not_none_op(self, to: int) -> None:
        item = self.pop()
        if item is not None:
            self.jump_forward_op(to)

    def build_slice_op(self, count: int) -> None:
        if count == 2:
            a, b = self.popn(2)
            self.push(slice(a, b))
        elif count == 3:
            a, b, step = self.popn(3)
            self.push(slice(a, b, step))

    def binary_subscr_op(self, arg: str) -> None:
        key = self.pop()
        container = self.pop()
        self.push(container[key])

    def build_list_op(self, count: int) -> None:
        self.push(list(self.popn(count)))

    def store_subscr_op(self, nothing: int) -> None:
        value, collection, key = self.popn(3)
        collection[key] = value

    def delete_subscr_op(self, nothing: int) -> None:
        collection, key = self.popn(2)
        del collection[key]

    def list_extend_op(self, i: int) -> None:
        iterable = self.pop()
        list.extend(self.data_stack[-i], iterable)

    def build_const_key_map_op(self, count: int) -> None:
        data = self.popn(count + 1)
        key_tuple = data[-1]
        item_map = dict()
        for i in range(count):
            item_map[key_tuple[i]] = data[i]
        self.push(item_map)

    def build_set_op(self, count: int) -> None:
        values = self.popn(count)
        self.push(set(values))

    def set_update_op(self, i: int) -> None:
        iterable = self.pop()
        set.update(self.data_stack[-i], iterable)

    def dict_update_op(self, i: int) -> None:
        iterable = self.pop()
        dict.update(self.data_stack[-i], iterable)

    def dict_merge_op(self, i: int) -> None:
        iterable = self.pop()
        dict.update(self.data_stack[-i], iterable)

    def format_value_op(self, flags: int) -> None:
        obj = self.pop()
        if (flags & 3) == 1:
            self.push(str(obj))
        elif (flags & 3) == 2:
            self.push(repr(obj))
        elif (flags & 3) == 3:
            self.push(ascii(obj))
        elif (flags & 4) == 1:
            self.push(self.pop()(obj))

    def build_string_op(self, count: int) -> None:
        self.push("".join(self.popn(count)))

    def unary_negative_op(self, nothing: int) -> None:
        self.push(-self.pop())

    def unary_positive_op(self, nothing: int) -> None:
        self.push(+self.pop())

    def unary_invert_op(self, nothing: int) -> None:
        self.push(~self.pop())

    def unary_not_op(self, nothing: int) -> None:
        self.data_stack[-1] = not self.top()

    def is_op(self, invert: int) -> None:
        left, right = self.popn(2)
        if invert:
            self.push(left is not right)
        else:
            self.push(left is right)

    def load_assertion_error_op(self, nothing: int) -> None:
        self.push(AssertionError)

    def _raise_exception(self, exception: tp.Any) -> None:
        self.last_raised_exception = exception
        raise exception

    def raise_varargs_op(self, argc: int) -> None:
        if argc == 0:
            raise ValueError
        elif argc == 1:
            exception = self.pop()
            self.last_raised_exception = exception
            raise exception
        elif argc == 2:
            exception, cause = self.popn(2)
            exception.__cause__ = cause
            self.last_raised_exception = exception
            raise exception

        raise ValueError

    def nop_op(self, nothing: int) -> None:
        pass

    def list_append_op(self, i: int) -> None:
        value = self.pop()
        list.append(self.data_stack[-i], value)

    def list_to_tuple_op(self) -> None:
        self.push(tuple(self.pop()))

    def build_map_op(self, count: int) -> None:
        data = self.popn(2 * count)
        item_map = dict()
        for i in range(0, 2 * count, 2):
            item_map[i] = data[i + 1]
        self.push(item_map)

    def map_add_op(self, i: int) -> None:
        val = self.pop()
        key = self.pop()
        self.data_stack[-i][key] = val

    def set_add_op(self, i: int) -> None:
        value = self.pop()
        set.add(self.data_stack[-i], value)

    def copy_op(self, nothing: int) -> None:
        self.push(self.top())

    def load_method_op(self, argval: str) -> None:
        value = self.pop()
        self.push_null_op(0)
        self.push(getattr(value, argval))

    def build_tuple_op(self, count: int) -> None:
        values = self.popn(count)
        self.push(values)

    def contains_op_op(self, invert: int) -> None:
        query, container = self.popn(2)
        ans: bool
        if hasattr(container, '__contains__'):
            ans = bool(container.__contains__(query))
        else:
            iterator = iter(container)
            query_found = False
            while True:
                if next(iterator) == query:
                    query_found = True
                    break
            ans = query_found
        self.push(ans if not invert else not ans)

    def import_name_op(self, name: str) -> None:
        level, fromlist = self.popn(2)
        self.push(
            __import__(name, self.globals, self.locals, fromlist, level)
        )

    def import_star_op(self, arg: tp.Any) -> None:
        # TODO: this doesn't use __all__ properly.
        md = self.pop()
        for attr in dir(md):
            if not attr.startswith('_'):
                self.locals[attr] = getattr(md, attr)

    def import_from_op(self, name: str) -> None:
        self.push(getattr(self.top(), name))

    def load_attr_op(self, attr: tp.Any) -> None:
        obj = self.pop()
        val = getattr(obj, attr)
        self.push(val)

    def store_attr_op(self, name: str) -> None:
        val, obj = self.popn(2)
        setattr(obj, name, val)

    def delete_attr_op(self, name: str) -> None:
        obj = self.pop()
        delattr(obj, name)

    def delete_global_op(self, name: str) -> None:
        del self.globals[name]

    def call_function_kw_op(self, arg: int) -> None:
        kwargs_keys = self.pop()
        val = self.popn(arg)
        func = self.pop()
        args: tp.Tuple[tp.Any, ...] = tuple(val[:arg - len(kwargs_keys)])
        kwargs: tp.Dict[tp.Any, tp.Any] = {}
        for ind in range(len(kwargs_keys)):
            kwargs[kwargs_keys[ind]] = val[ind + arg - len(kwargs_keys)]
        self.push(func(*args, **kwargs))

    def call_function_var_op(self, arg: int) -> None:
        args = self.pop()
        f = self.pop()
        self.push(f(args, {}))

    def delete_fast_op(self, name: str) -> None:
        del self.locals[name]

    def swap_op(self, i: int) -> None:
        item = self.data_stack[i - 1]
        tos = self.pop()
        try:
            self.data_stack[i - 1] = tos
        except Exception:
            self.push(tos)
        self.push(item)

    def extended_arg_op(self, ext: int) -> None:
        pass


class VirtualMachine:
    def run(self, code_obj: types.CodeType) -> None:
        """
        :param code_obj: code for interpreting
        """
        globals_context: dict[str, tp.Any] = {}
        frame = Frame(code_obj, builtins.globals()['__builtins__'],
                      globals_context, globals_context)
        return frame.run()
