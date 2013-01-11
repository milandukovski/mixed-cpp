#include <iostream>
#include <vector>

using namespace std;


//----------------------------- Example 1
#include <boost/format.hpp>

void example_1()
{
    cout << boost::format("I like %2% more than %1%! \n") % 42 % 17;
}

//----------------------------- Example 2

void example_2()
{
    vector<vector<float>> verts = {{0,0,0}, {1,0,0}, {0,1,0}};

    for(auto vert : verts)
    {
        cout << "( ";
        for(auto comp : vert)
            cout << comp << ' ';
        cout << ')' << endl;
    }
}

//----------------------------- Example 3
#include <array>

void example_3()
{
    array <float, 5> a{{1,2,3}};

    for(auto i : a)
        cout << i << endl;

    cout << "---" << endl;

    int n = 0;
    for(auto& i : a)
    {
        i = n = n + 2;
        cout << i << endl;
    }
}

//----------------------------- Example 4

class Shader
{
    int next_id = 0;

    private:
        string vertex_path;
        string fragment_path;
        int id;

    public:
        Shader(const string& vpath, const string& fpath, int id)
            : vertex_path(vpath), 
              fragment_path(fpath),
              id(id)
              {}

        const string&   get_vertex_path()       const { return vertex_path;     };
        const string&   get_fragment_path()     const { return fragment_path;   };
        int             get_id()                const { return id;              };
};

void example_4()
{
    vector<Shader> materials;
    materials.push_back(Shader("phong.vs",  "phong.fs",  3));
    materials.push_back(Shader("blinn.vs",  "blinn.fs",  2));
    materials.push_back(Shader("xray.vs",   "xray.fs",   1));

    cout << endl << "pre sort:" << endl;
    for(auto m : materials)
        cout << m.get_vertex_path() << " :: " << m.get_id() << endl;

    sort(begin(materials), end(materials), 
         [](const Shader& s1, const Shader& s2) {
            return s1.get_id() < s2.get_id();
         });

    cout << endl << "post sort:" << endl;
    for(auto m : materials)
        cout << m.get_vertex_path() << " :: " << m.get_id() << endl;
}

//----------------------------- Example 5

void _nullptr_overload(nullptr_t)
{
    cout << "argument is a nullptr!" << endl;
}

void example_5()
{
    _nullptr_overload(nullptr);
}

//----------------------------- Example 6

template <typename T>
void show(const T& value)
{
    std::cout << value << endl;
}

template <typename T, typename... U>
void show(const T& head, const U&... tail)
{
    cout << head << ' ';
    show(tail...);
}

void example_6()
{
    show(17, 1.5, "Ahoi!");
}

//----------------------------- Example 7
#include <unordered_map>

struct obj_id
{
    string path;
    int id;

    obj_id(const string& p, int id)
        : path(p),
          id(id)
          {}
};

ostream& operator<<(ostream& os, const obj_id& o)
{
    return os << "OBJ :: " << o.id << " (" << o.path << ")";
}

class obj_hash
{
    public:
        long operator()(const obj_id& o) const
        {
            hash<string> h;
            return h(o.path) * 100 + o.id;
        }
};

class obj_equal_to
{
    public:
        bool operator()(const obj_id& a, const obj_id& b) const
        {
            return a.path == b.path && a.id == b.id;
        }
};

class Obj
{
    string path;
    vector<array<float, 3>> vertices;
    vector<array<float, 3>> normals;
    vector<array<float, 2>> texcoords;


    public:
        Obj(const string& path)
            : path(path)
              {
                    // Cube
                    vertices = {{{-0.5, 0.5,0.0}},
                                {{ 0.5, 0.5,0.0}},
                                {{ 0.5,-0.5,0.0}},
                                {{-0.5,-0.5,0.0}}};
              }

    const string& get_path() const { return path; };
    vector<array<float, 3>> get_vertices() const { return vertices; };
};

ostream& operator<<(ostream& os, const Obj& o)
{
    for(auto vert : o.get_vertices())
    {
        os << '(';
        for(auto i : vert)
            os << i << ' ';
        os << ')' << ",  " ;
    }
    return os;
}

void example_7()
{
    unordered_map<obj_id, Obj, obj_hash, obj_equal_to> um;
    um.insert(make_pair( obj_id("teapot.obj", 1), Obj("teapot.obj") ));
    um.insert(make_pair( obj_id("troll.obj",  2), Obj("troll.obj" ) ));

    for(const auto& item : um)
        cout << item.first << ":  " << item.second << endl;
}

//----------------------------- Example 8

void _just_a_function()
{
    cout << "Function here!" << endl;
}

class _functor
{
    public:
        void operator()() const
        {
            cout << "Functor here!" << endl;
        }
};

void _execute(const vector<function<void()>>& fs)
{
    for (auto& f : fs)
        f();
}

void example_8()
{
    vector<function<void()>> tests;

    tests.push_back(_just_a_function);
    _functor func;
    tests.push_back(func);
    tests.push_back([](){ cout << "Lambda here!" << endl; } );

    _execute(tests);
}

//----------------------------- Example 9
#include <type_traits>

struct is_64_bit
{
    static const bool value = sizeof(void*) == 8;
};

enable_if<is_64_bit::value, void>::type
my_memcpy(void* target, const void* source, size_t n)
{
    cout << "64 bit memcpy" << endl;
}

void example_9()
{
    int a = 17, b = 42;
    void* ap = &a;
    void* bp = &b;

    my_memcpy(ap, bp, 10);
}

//----------------------------- Example 10
#include <functional>

int mult(int a, int b)
{
    return a * b;
}


void add(int a, int b, int& r)
{
    r = a + b;
}

class A {};
class B : public A {};

void example_10()
{
    int result = 0;
    auto f2 = bind(add, placeholders::_2, placeholders::_1, ref(result));
    f2(17, 42);
    cout << result << endl;

    A a, c;
    B b, d;

    vector<reference_wrapper<A>> v = {a, b, c, d};

    auto f = bind(mult, 2, placeholders::_1);
    for (int i : {1,2,3,4,5,6,7,8,9})
        cout << "2 * " << i << " = " << f(i) << endl; 

}

//----------------------------- Example 11

class _NoAggregate
{
    int x;                      // private by default
    virtual void f(){};         // virtual method

    public:
        _NoAggregate(int) {}    // user-defined constructor
};

class _Aggregate1
{
    public:
        _NoAggregate n;
        _Aggregate1& operator= (_Aggregate1 const & rhs) 
        {
            // copy-assignement - ...
            return (_Aggregate1&) rhs;
        }
    private:
        void some_private_function() {}             
};

class _Agg1
{
    public:
        int a;
        _Agg1() {                   // default constructor available
            a = 17;
        }
};

struct _Agg2
{
    int i1;
    int i2;
};

struct _Agg3
{
    char c;
    _Agg2 x;
    int i[2];
    float f;

    protected:
        static double d;            
    private:
        void g(){}
};

void example_11()
{

    _Agg1 agg[3] = {_Agg1()};           // b[1,2] are value initialized 
    int a[3] = {1};                     // the first is 1, the rest is 0
    _Agg3 y = {'a', {10,20}, {30,40}};  // .f=0.0  ,  .d will not be initialized


    cout << agg[0].a << " -- " << agg[1].a << endl;
    cout << a[0] << " -- " << a[1] << endl;
    cout << y.c << " | " << y.x.i1 << " | " << y.x.i2 << " | " << y.i[0] <<  " | " << y.i[1] << " | " << y.f << endl;

}

//----------------------------- Example 12

struct _POD
{
    int i;
    char c;
    void f() {}
    static vector<char> v;
};

struct _AggButNoPOD1
{
    int x;
    ~_AggButNoPOD1(){}          // user-defined destructor
};

struct _AggButNoPOD2
{
    _AggButNoPOD1 a[3];         // array of non-POD
};

void example_12()
{
    char buf[sizeof(_POD)];
    _POD obj = {9, 'a'};

    cout << obj.i << "  " << obj.c << endl;

    memcpy(buf, &obj, sizeof(_POD)); 

    obj.i = 42;
    obj.c = 'c';

    cout << obj.i << "  " << obj.c << endl;

    memcpy(&obj, buf, sizeof(_POD)); 

    cout << obj.i << "  " << obj.c << endl;
}

//----------------------------- Example 13

/* The new definition basically says that 
   a POD is a class that is both trivial 
   and has standard-layout, and this 
   property must hold recursively for all
   non-static data members. */


void example_13() {}
//----------------------------- Example 14
void example_14() {}
//----------------------------- Example 15
void example_15() {}



//----------------------------- MAIN

int main()
{
    std::vector<function<void()>> examples = {{ 
                                                example_1, example_2, example_3, example_4, example_5, 
                                                example_6, example_7, example_8, example_9, example_10,
                                                example_11, example_12, example_13, example_14, example_15, 
                                             }};

    cout << endl;

    int n = 0;
    for(auto& e : examples)
    {
        cout << endl << "-------------------  Example " 
             << ++n << endl << endl;
        e();
    }


    cout << endl;
    return 0;
}