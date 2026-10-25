#include <iostream>
#include <memory>
#include <vector>
#include <functional>
#include <unordered_set>
#include <cmath>
#include <random>
#include <chrono>
#include <iterator>
#include <sstream>
#include <cassert>


class Value;

using ValuePtr = std::shared_ptr<Value>;

struct Hash{
	size_t operator()(const ValuePtr value) const;
};

class Value: public std::enable_shared_from_this<Value>{

	public:
		inline static size_t currentID;
		float data;
		float grad = 0.0f;
		std::string op;
		size_t id;
		std::vector<ValuePtr> prev;
		std::function<void()> backward;
		float velocity = 0.0f;
		float r = 0.0f;
		float m = 0.0f;
		
	private:
		Value(float data, const std::string &op, size_t id) : data(data) , op(op), id(id) {};	
	public:
		static ValuePtr create(float data, const std::string &op = ""){
			return std::shared_ptr<Value>(new Value(data, op, Value::currentID++));
		}
		
		~Value(){
			//Value::currentID--;
		}
		
	static ValuePtr add(const ValuePtr &lhs, const ValuePtr &rhs){
		auto out = Value::create(lhs->data + rhs->data, "+");
		out->prev = {lhs, rhs};
		out->backward = [lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)](){
			lhs_weak.lock()->grad += out_weak.lock()->grad;
			rhs_weak.lock()->grad += out_weak.lock()->grad;
		};
		return out;
	}
	
	static ValuePtr substract(const ValuePtr &lhs, const ValuePtr &rhs){
		auto out = Value::create(lhs->data - rhs->data, "-");
		out->prev = {lhs, rhs};
		out->backward = [lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)](){
			lhs_weak.lock()->grad += out_weak.lock()->grad;
			rhs_weak.lock()->grad -= out_weak.lock()->grad; 
		};
		return out;
	}
	
	static ValuePtr multiply(const ValuePtr &lhs, const ValuePtr &rhs){
		auto out = Value::create(lhs->data * rhs->data, "*");
		out->prev = {lhs, rhs};
		out->backward = [lhs_weak = std::weak_ptr<Value>(lhs), rhs_weak = std::weak_ptr<Value>(rhs), out_weak = std::weak_ptr<Value>(out)](){
			lhs_weak.lock()->grad += out_weak.lock()->grad * rhs_weak.lock()->data;
			rhs_weak.lock()->grad += out_weak.lock()->grad * lhs_weak.lock()->data;
		};
		return out;
	}
	
	static ValuePtr pow(const ValuePtr &base, float exponent){
		auto out = Value::create(std::pow(base->data, exponent), "^");
		out->prev = {base};
		out->backward = [base_weak = std::weak_ptr<Value>(base), out_weak = std::weak_ptr<Value>(out), exponent](){
			base_weak.lock()->grad += out_weak.lock()->grad * exponent * std::pow(base_weak.lock()->data, exponent - 1);
		};
		return out;
	}
	
	static ValuePtr divide(const ValuePtr &lhs, const ValuePtr &rhs){
		auto reciprocal = pow(rhs, -1);
		return multiply(lhs, reciprocal);
		
	};
	
	
	static ValuePtr relu(const ValuePtr &input){
		float val = std::max(0.0f, input->data);
		auto out = Value::create(val, "ReLU");
		out->prev = {input};
		out->backward = [input_weak = std::weak_ptr(input), out_weak = std::weak_ptr(out)](){
			//out grad * inward gradient
			input_weak.lock()->grad += (out_weak.lock()->data > 0) * out_weak.lock()->grad;
		};
		return out;
	}
	
	static ValuePtr sigmoid(const ValuePtr & input){
		float d = input->data;
		float sig = 1 / (1 + std::exp(-d));
		auto out = Value::create(sig, "Sigmoid");
		out->prev = {input};
		out->backward = [input_weak = std::weak_ptr(input), out_weak = std::weak_ptr(out), sig](){
			//out grad * inward gradient
			input_weak.lock()->grad += out_weak.lock()->grad * (sig * (1 - sig));
		};	
		return out;
	}
	
	
	void buildtopo(ValuePtr v, std::unordered_set<ValuePtr, Hash>& visited, std::vector<ValuePtr>& topo){
		if(visited.find(v) == visited.end()){
			visited.insert(v);
			for(const auto& child : v->prev)
				buildtopo(child, visited, topo);
			topo.push_back(v);
		}
	}
	
	void backprop(){
		std::unordered_set<ValuePtr, Hash> visited;
		std::vector<ValuePtr> topo;
		buildtopo(shared_from_this(), visited, topo);
		
		this->grad = 1.0f;
		
		for(auto it = topo.rbegin(); it != topo.rend(); it++){
			if((*it)->backward)
				(*it)->backward();
			//(*it)->print();
		}
	}
	
	void print(){
		std::cout << "[Data: " << this->data << "\t Grad: " << this->grad << "]" << std::endl;
	}
};

size_t Hash::operator()(const ValuePtr value) const{
	return std::hash<std::string>()(value->op) ^ std::hash<float>()(value->id);
}


enum ActivationType{
	RELU,
	SIGMOID
};

class Activation{
	public:
		static ValuePtr Relu(const ValuePtr &input){
			return Value::relu(input);
		}
		
		static ValuePtr Sigmoid(const ValuePtr &input){
			return Value::sigmoid(input);
		}
		
		static inline std::unordered_map<ActivationType, std::function<ValuePtr(const ValuePtr&)>> actfunc{
			{ActivationType::RELU, Relu},
			{ActivationType::SIGMOID, Sigmoid}
		};
};

template<typename T>
std::string printVector(const std::vector<T>& vec){
	std::ostringstream oss;
	oss << "[";
	for(size_t idx = 0; idx < vec.size(); ++idx){
		oss << vec[idx] << (idx == vec.size() - 1 ? "" : ", ");
	}
	oss << "]";
	return oss.str();
}

template<typename T>
std::string printVectorData(const std::vector<T>& vec){
	std::ostringstream oss;
	oss << "[";
	for(size_t idx = 0; idx < vec.size(); ++idx){
		oss << vec[idx]->data << (idx == vec.size() - 1 ? "" : ", ");
	}
	oss << "]";
	return oss.str();
}

class Tensor{
	private:
		std::vector<ValuePtr> tensor;
		std::vector<size_t> shape;
		std::vector<size_t> stride;
		size_t ne = 0;
		
	public:
		Tensor(std::vector<ValuePtr> data, std::vector<size_t> shape) : shape(shape){
			if(shape.empty())
				throw std::invalid_argument("Shape cannot be empty");
			
			ne = 1;
			for(auto& s : shape){
				if(s == 0)
					throw std::invalid_argument("A dimension can not be 0");
				ne *= s;
			}
				
			if (data.size() != ne)
				throw std::invalid_argument("Either excess or deficient elements in the tensor");
			
			tensor = data;
			
			stride.resize(shape.size());
			
			stride[shape.size() - 1] = 1;
			
			if(shape.size() > 1)
				for(size_t idx = shape.size() - 1; idx > 0; idx--)
					stride[idx - 1] = stride[idx] * shape[idx];
		}
		
		Tensor(std::vector<ValuePtr> data, std::vector<size_t> shape, std::vector<size_t> stride) : tensor(data), shape(shape), stride(stride){
			if(shape.empty())
				throw std::invalid_argument("Shape cannot be empty");
			
			if(stride.empty())
				throw std::invalid_argument("Stride cannot be empty");
			
			if(shape.size() != stride.size())
				throw std::invalid_argument("Shape and Stride mismatch");
			
			ne = 1;
			for(auto& s : shape){
				if(s == 0)
					throw std::invalid_argument("A dimension can not be 0");
				ne *= s;
			}
				
			if (data.size() != ne)
				throw std::invalid_argument("Either excess or deficient elements in the tensor");
		}
		
		void backprop(){
			for(const auto &v : data())
				v->backprop();
		}
		
		ValuePtr& at(std::vector<size_t> pos){
			size_t offset = 0;
			if(pos.size() != shape.size())
				throw std::invalid_argument("Cannot access non-existent dimensions");
			
			for(size_t idx = 0; idx < pos.size(); ++idx){
				if(pos[idx] >= shape[idx])
					throw std::out_of_range("Index " + std::to_string(idx) + " is out of range");
				
				offset += pos[idx] * stride[idx];
			}
			
			return tensor[offset];
		}
		
		const ValuePtr& at(std::vector<size_t> pos) const{
			size_t offset = 0;
			if(pos.size() != shape.size())
				throw std::invalid_argument("Cannot access non-existent dimensions");
			
			for(size_t idx = 0; idx < pos.size(); ++idx){
				if(pos[idx] >= shape[idx])
					throw std::out_of_range("Index " + std::to_string(idx) + " is out of range");
				
				offset += pos[idx] * stride[idx];
			}
			
			return tensor[offset];
		}
		
		Tensor getSample(size_t sampleIndex) const{
			if(sampleIndex >= shape[0])
        		throw std::invalid_argument("Cannot access elements outside the sample");

			size_t slice = stride[0];
	
			std::vector<ValuePtr> data;
   		data.reserve(slice);

			size_t offset = sampleIndex * slice;
	
			for(size_t idx = 0; idx < slice; ++idx)
				data.push_back(tensor[offset + idx]);
	
			std::vector<size_t> sample_shape(shape.begin() + 1, shape.end());
	
			return Tensor(data, sample_shape);
		}
		
		Tensor slice(std::vector<size_t> sliceIndex) const{
			if(sliceIndex.empty())
				throw std::invalid_argument("The slice input cannot be empty");
			
			if(sliceIndex.size() >= shape.size())
				throw std::invalid_argument("Too many indices");
			
			for(size_t idx = 0; idx < sliceIndex.size(); ++idx)
				if(sliceIndex[idx] >= shape[idx])
					throw std::out_of_range("Cannot access elements outside the sample");
					
			size_t offset = 0;
			
			for(size_t idx = 0; idx < sliceIndex.size(); ++idx)
				offset += stride[idx] * sliceIndex[idx];
			
			size_t slice = stride[sliceIndex.size() - 1];
			
			std::vector<ValuePtr> data;
   		data.reserve(slice);
			
			for(size_t idx = 0; idx < slice; ++idx)
				data.push_back(tensor[offset + idx]);
			
			std::vector<size_t> sample_shape(shape.begin() + sliceIndex.size(), shape.end());
	
			return Tensor(data, sample_shape);
			
		}
		
		Tensor reshape(std::vector<size_t> newShape){
			if(newShape.empty())
			    throw std::invalid_argument("Shape cannot be empty");
			
			size_t numel = 1;
			for(size_t s : newShape){
				if(s == 0)
					throw std::invalid_argument("A dimension can not be 0");
				numel *= s;
			}
			
			if(ne != numel)
				throw std::invalid_argument("The new shape is invalid");
			
			return Tensor(tensor, newShape);
		}
		
		Tensor transpose_2d() const{
			if(ndim() != 2)
				throw std::invalid_argument("transpose_2d only supports 2d vectors");
				
			std::vector<size_t> yshape = {getShape()[1], getShape()[0]};
			std::vector<size_t> ystride = {getStride()[1], getStride()[0]};
			
			return Tensor(data(), yshape, ystride);
		}
		
		bool iscontiguous() const{
			std::vector<size_t> stride;
		
			stride.resize(ndim());
			
			stride[shape.size() - 1] = 1;
			
			if(shape.size() > 1)
				for(size_t idx = shape.size() - 1; idx > 0; idx--)
					stride[idx - 1] = stride[idx] * shape[idx];
			
			return (this->stride == stride);
		}
		
		Tensor contiguous() const{
			if(iscontiguous())
				return *this;
			
			std::vector<ValuePtr> cont;
			cont.reserve(numel());
			
			for(size_t i = 0; i < numel(); ++i){
				cont.push_back(at(flat_to_multidimension(i)));
			}
			
			return Tensor(cont, getShape());
		}
		
		const std::vector<ValuePtr>& data() const{
			return tensor;
		}
		
		const std::vector<size_t>& getShape() const{
			return shape;
		}
		
		const std::vector<size_t>& getStride() const{
			return stride;
		}
		
		size_t numel() const{
			return ne;
		}
		
		size_t ndim() const{
			return shape.size();
		}
		
		static std::vector<size_t> broadcastShape(std::vector<size_t> a, std::vector<size_t> b){
			size_t maxSize = std::max(a.size(), b.size());
			
			std::vector<size_t> brdcstShape(maxSize);
			
			for(size_t i = 0; i < maxSize; ++i){
				size_t x = (i < a.size()) ? a[a.size() - i - 1] : 1;
				size_t y = (i < b.size()) ? b[b.size() - i - 1] : 1;
				
				if(x == y || ((x == 1) || (y == 1)))
					brdcstShape[brdcstShape.size() - i - 1] = std::max(x, y);
				else
					throw std::invalid_argument("Broadcasting shapes are incompatible");
			}
			return brdcstShape;
		}
		
		std::vector<size_t> broadcastIndexMap(std::vector<size_t> ip) const{
			//std::cout << "\nOutput tensor size: " << getShape().size() << "\nInput Tensor size" << ip.size() << std::endl;
			if(getShape().size() > ip.size())
				//std::cout << "Mapping Tensor\n Shape: " << ip.getShape() << "Index">>
				throw std::invalid_argument("Broadcast index mapping cannot continue due to output tensor size being greater than it's broadcast tensor size");
			size_t ds = ip.size() - getShape().size();
			std::vector<size_t> op(getShape().size());
			for(size_t idx = 0; idx < ip.size(); ++idx){
				if(idx < ds)
					continue;
				
				op[idx - ds] = (getShape()[idx - ds] == 1) ? 0 : ip[idx];
			}
			return op;
		}
		
		static std::vector<size_t> broadcastIndexMap(std::vector<size_t> shape,std::vector<size_t> ip){
			//std::cout << "\nOutput tensor size: " << getShape().size() << "\nInput Tensor size" << ip.size() << std::endl;
			if(shape.size() > ip.size())
				//std::cout << "Mapping Tensor\n Shape: " << ip.getShape() << "Index">>
				throw std::invalid_argument("Broadcast index mapping cannot continue due to output tensor size being greater than it's broadcast tensor size");
			size_t ds = ip.size() - shape.size();
			std::vector<size_t> op(shape.size());
			for(size_t idx = 0; idx < ip.size(); ++idx){
				if(idx < ds)
					continue;
				
				op[idx - ds] = (shape[idx - ds] == 1) ? 0 : ip[idx];
			}
			return op;
		}
		
		Tensor broadcastAdd(Tensor other) const{
			std::vector<size_t> shape = broadcastShape(getShape(), other.getShape());
			
			size_t numel = 1;
			for(size_t s : shape)
				numel *= s;
			
			std::vector<ValuePtr> out;
			out.reserve(numel);
			
			std::vector<size_t> index(shape.size());
			for(size_t i = 0; i < numel; i++){
				size_t temp = i;
				for(size_t j = shape.size(); j-- > 0;){
					index[j] = temp % shape[j];
					temp = temp / shape[j];
				}
				ValuePtr lhs = at(broadcastIndexMap(index));
				ValuePtr rhs = other.at(other.broadcastIndexMap(index));
				out.push_back(Value::add(lhs, rhs));
			}
			
			return Tensor(out, shape);
		}
		
		Tensor broadcastAddExp(Tensor other) const{
			return flat_elementwise_operation(Value::add, *this, other);
		}
		
		static size_t numel(std::vector<size_t> shape){
			size_t numel = 1;
			for(size_t s : shape)
				numel *= s;
			return numel;
		}
		
		std::vector<size_t> flat_to_multidimension(size_t& i) const{
			std::vector<size_t> index(ndim());
			size_t temp = i;
			for(size_t j = getShape().size(); j-- > 0;){
				index[j] = temp % getShape()[j];
				temp = temp / getShape()[j];
			}
			return index;
		}
		
		static std::vector<size_t> flat_to_multidimension(size_t& i, std::vector<size_t>& shape, std::vector<size_t> out_index){
			size_t temp = i;
			for(size_t j = shape.size(); j-- > 0;){
				out_index[j] = temp % shape[j];
				temp = temp / shape[j];
			}
			return out_index;
		}
		
		template<typename op>
		static Tensor flat_elementwise_operation(op func, const Tensor& lhs, const Tensor& rhs){
			std::vector<ValuePtr> out;
			std::vector<size_t> shape;
			
			if(lhs.getShape() == rhs.getShape()){
				shape = lhs.getShape();
				out.reserve(numel(shape));
				for(size_t i = 0; i < numel(shape); i++)
					out.push_back(func(lhs.data()[i], rhs.data()[i]));
				return Tensor(out, shape);
			}
					
			shape = broadcastShape(lhs.getShape(), rhs.getShape());
			out.reserve(numel(shape));
			std::vector<size_t> index(shape.size());
			for(size_t i = 0; i < numel(shape); i++){
				index = flat_to_multidimension(i, shape, index);
				ValuePtr ls = lhs.at(lhs.broadcastIndexMap(index));
				ValuePtr rs = rhs.at(rhs.broadcastIndexMap(index));
				out.push_back(func(ls, rs));
			}
			return Tensor(out, shape);
		}
		
		friend Tensor operator+(const Tensor& lhs, const Tensor& rhs){
			return flat_elementwise_operation(Value::add, lhs, rhs);
		}
		
		friend Tensor operator-(const Tensor& lhs, const Tensor& rhs){
			return flat_elementwise_operation(Value::substract, lhs, rhs);
		}
		
		friend Tensor operator*(const Tensor& lhs, const Tensor& rhs){
			return flat_elementwise_operation(Value::multiply, lhs, rhs);
		}
		
		friend Tensor operator/(const Tensor& lhs, const Tensor& rhs){
			return flat_elementwise_operation(Value::divide, lhs, rhs);
		}
		
		Tensor matmul_2d(const Tensor& other) const{
			std::vector<size_t> lstride = getStride();
			std::vector<size_t> rstride = other.getStride();
			
			if(((ndim() != 2) || (other.ndim() != 2)) || (getShape()[1]) != other.getShape()[0])
				throw std::invalid_argument("Conditions for Matrix multiplication 2D aren't met.");
			
			std::vector<size_t> shape = {getShape()[0], other.getShape()[1]};
			std::vector<ValuePtr> op;
			op.reserve(numel(shape));
			
			for(size_t i = 0; i < getShape()[0]; i++)
				for(size_t j = 0; j < other.getShape()[1]; j++){
					ValuePtr sum = nullptr;
					for(size_t k = 0; k < getShape()[1]; k++){
						ValuePtr prod = Value::multiply(data()[(i * lstride[0]) + k], other.data()[j + (k * rstride[0])]);
						if(sum == nullptr)
							sum = prod;
						else
							sum = Value::add(sum, prod);
					}
					op.push_back(sum);
				}
			return Tensor(op, shape);
		}
		
		Tensor bmm_3d(const Tensor& other) const{
			if(ndim() != 3 || other.ndim() != 3)
				throw std::invalid_argument("bmm only handles 3 dimensions");
				
			if(getShape()[0] != other.getShape()[0])
				throw std::invalid_argument("Batch shapes must be the same");
			
			if(getShape()[2] != other.getShape()[1])
				throw std::invalid_argument("Inner matrix dimensions do no align for bmm");
				
			size_t B = getShape()[0];
			size_t M = getShape()[1];
			size_t K = getShape()[2];
			size_t N = other.getShape()[2];
			
			std::vector<size_t> outShape = {B, M, N};
			std::vector<ValuePtr> out_data;
			out_data.reserve(B * M * N);
			
			size_t bstride1 = getStride()[0];
			size_t bstride2 = other.getStride()[0];
			size_t mstride = getStride()[1];//row stride of 1st matrix
			size_t mk_stride = getStride()[2];//column stride of 1st matrix
			size_t nstride = other.getStride()[1];//row stride of 2nd matrix
			size_t nk_stride = other.getStride()[2];//column stride of 2nd matrix
			
			const auto& d1 = data();
			const auto& d2 = other.data();
			
			for(size_t b = 0; b < B; ++b){
				size_t b_off1 = b * bstride1;
				size_t b_off2 = b * bstride2;
				for(size_t i = 0; i < M; i++){
					size_t row_off1 = b_off1 + (i * mstride);
					for(size_t j = 0; j < N; j++){
						size_t col_off2 = b_off2 + (j * nk_stride);
					
						ValuePtr sum = nullptr;
						for(size_t k = 0; k < K; k++){
							//ValuePtr prod = Value::multiply(data()[(b * bstride1) + (i * mstride) + (k * mk_stride)], other.data()[(b * bstride2) + ((j * nk_stride) + (k * nstride))]);
							ValuePtr prod = Value::multiply(
								d1[row_off1 + (k * mk_stride)],
								d2[col_off2 + (k * nstride)]
							);
							
							if(sum == nullptr)
								sum = prod;
							else
								sum = Value::add(sum, prod);
						}
						out_data.push_back(sum);
					}
				}
			}
			return Tensor(out_data, outShape);
		}
		
		Tensor unsqueeze(size_t dim) const{
			if(dim > ndim())
				throw std::invalid_argument("Unsqueeze dim out of range");
			
			std::vector<size_t> nshape = getShape();
			nshape.insert(nshape.begin() + dim, 1);
			
			std::vector<size_t> nstride = getStride();
			size_t insert_stride = (dim < nstride.size()) ? nstride[dim] : 1;
			nstride.insert(nstride.begin() + dim, insert_stride);
			
			return Tensor(data(), nshape, nstride);
		}
		
		Tensor squeeze(size_t dim) const{
			if(dim >= ndim())
				throw std::invalid_argument("Squeeze dim out of range");
			
			if(getShape()[dim] != 1)
				throw std::invalid_argument("Cannot squeeze dim with shape != 1");
				
			std::vector<size_t> nshape = getShape();
			std::vector<size_t> nstride = getStride();
			
			nshape.erase(nshape.begin() + dim);
			nstride.erase(nstride.begin() + dim);
			
			return Tensor(data(), nshape, nstride);
		}
		
		Tensor matmul(const Tensor& other) const{
			size_t rank1 = ndim();
			size_t rank2 = other.ndim();
			
			if(rank1 == 0 || rank2 == 0)
				throw std::invalid_argument("0 dimensional matmul is not allowed");
			
			if((rank1 == 1) && (rank2 == 1)){
				if(getShape()[0] != other.getShape()[0])
					throw std::invalid_argument("1D matrix multiplication must have same same lengths for both matrices");	
				
				int K = getShape()[0];
				
				ValuePtr sum = nullptr;
				for(size_t k = 0; k < K; k++){
					//ValuePtr prod = Value::multiply(data()[(b * bstride1) + (i * mstride) + (k * mk_stride)], other.data()[(b * bstride2) + ((j * nk_stride) + (k * nstride))]);
					ValuePtr prod = Value::multiply(
						data()[(k * getStride()[0])],
						other.data()[(k * other.getStride()[0])]
					);
					sum = (sum) ? Value::add(sum, prod) : prod;
				}
				return Tensor({sum}, {1});
			}
			
			if((rank1 == 1) && (rank2 >= 2)){
				Tensor x = unsqueeze(0);
				Tensor res = x.matmul(other);
				return res.squeeze(res.ndim() - 2);
			}
			
			if((rank1 >= 2) && (rank2 == 1)){
				Tensor x = other.unsqueeze(1);
				Tensor res = this->matmul(x);
				return res.squeeze(res.ndim() - 1);
			}
			
			if((rank1 == 2) && (rank2 == 2)){
				return this->matmul_2d(other);
			}
			
			std::vector<size_t> batch_shape1(getShape().begin(), getShape().end() - 2);
			std::vector<size_t> batch_shape2(other.getShape().begin(), other.getShape().end() - 2);
			
			std::vector<size_t> brd_batch_shape = broadcastShape(batch_shape1, batch_shape2);
			
			size_t tot_bat = numel(brd_batch_shape);
			
			if(getShape()[rank1 - 1] != other.getShape()[rank2 - 2])
				throw std::invalid_argument("Inner matrix dimensions do no align for matmul");
			
			size_t M = getShape()[rank1 - 2];
			size_t K = getShape()[rank1 - 1];
			size_t N = other.getShape()[rank2 - 1];
			
			std::vector<size_t> outShape = brd_batch_shape;
			outShape.push_back(M);
			outShape.push_back(N);
			
			std::vector<ValuePtr> out_data;
			out_data.reserve(numel(outShape));
			
			const auto& d1 = data();
			const auto& d2 = other.data();
			
			size_t mstride = getStride()[rank1 - 2];//row stride of 1st matrix
			size_t mk_stride = getStride()[rank1 - 1];//column stride of 1st matrix
			size_t nstride = other.getStride()[rank2 - 2];//row stride of 2nd matrix
			size_t nk_stride = other.getStride()[rank2 - 1];//column stride of 2nd matrix
			
			std::vector<size_t> b1_index;
			b1_index.reserve(batch_shape1.size());
			std::vector<size_t> b2_index;
			b2_index.reserve(batch_shape2.size());
			
			std::vector<size_t> op_batch_idx(brd_batch_shape.size(), 0);
			
			for(size_t b = 0; b < tot_bat; ++b){
				size_t b_off1 = 0;
				size_t b_off2 = 0;
				
				op_batch_idx = flat_to_multidimension(b, brd_batch_shape, op_batch_idx);
				
				b1_index = broadcastIndexMap(batch_shape1, op_batch_idx);
				b2_index = broadcastIndexMap(batch_shape2, op_batch_idx);
				
				for(size_t d = 0; d < batch_shape1.size(); ++d)
					b_off1 += (b1_index[d] * getStride()[d]);
				for(size_t d = 0; d < batch_shape2.size(); ++d)
					b_off2 += (b2_index[d] * other.getStride()[d]);
				
				for(size_t i = 0; i < M; i++){
					size_t row_off1 = b_off1 + (i * mstride);
					for(size_t j = 0; j < N; j++){
						size_t col_off2 = b_off2 + (j * nk_stride);
					
						ValuePtr sum = nullptr;
						for(size_t k = 0; k < K; k++){
							//ValuePtr prod = Value::multiply(data()[(b * bstride1) + (i * mstride) + (k * mk_stride)], other.data()[(b * bstride2) + ((j * nk_stride) + (k * nstride))]);
							ValuePtr prod = Value::multiply(
								d1[row_off1 + (k * mk_stride)],
								d2[col_off2 + (k * nstride)]
							);
							
							if(sum == nullptr)
								sum = prod;
							else
								sum = Value::add(sum, prod);
						}
						out_data.push_back(sum);
					}
				}
			}
			return Tensor(out_data, outShape);
		}
		
		Tensor sum() const{
			ValuePtr s = nullptr;
			for(ValuePtr v : data()){
				if(!s) s = v;	
				else s = Value::add(s, v);
			}
			
			return Tensor({s}, {1});
		}
		
		Tensor sum(size_t axis) const{
			if(axis >= ndim())
				throw std::invalid_argument("The axis given is more than the output dimension and therefore out of bounds");
			
			if(ndim() == 1)
				return sum();
		
			std::vector<size_t> outShape;
			outShape.reserve(ndim());
			for(size_t i = 0; i < ndim(); ++i)
				if(i != axis)
					outShape.push_back(getShape()[i]);
			
			std::vector<ValuePtr> sumVector;
			sumVector.reserve(numel(outShape));
			
			size_t totalOut = numel(outShape);
			std::vector<size_t> out_idx(outShape.size(), 0);
			std::vector<size_t> in_idx(ndim(), 0);
			
			for(size_t i = 0; i < totalOut; ++i){
				out_idx = flat_to_multidimension(i, outShape, out_idx);
				
				size_t ctr = 0;
				for(size_t d = 0; d < ndim(); ++d)
					if(d != axis)
						in_idx[d] = out_idx[ctr++];
				
				ValuePtr sum = nullptr;
				
				for(size_t j = 0; j < getShape()[axis]; ++j){
					in_idx[axis] = j;
					
					if(!sum) sum = at(in_idx);
					else sum = Value::add(sum, at(in_idx));
				}
				sumVector.push_back(sum);
			}
			return Tensor(sumVector, outShape);
		}
};

float getRandomFloat(){
	static std::random_device rd;
	static std::mt19937 rng(rd());
	static std::uniform_real_distribution dis(-1.0f,1.0f);
	return dis(rng);
}


static std::vector<ValuePtr> randomWeights(size_t nin){
	std::vector<ValuePtr> rnd;
	for(size_t in = 0; in < nin; in++)
		rnd.emplace_back(Value::create(getRandomFloat()));
	return rnd;
}

static std::vector<ValuePtr> zeroBiases(size_t nout){
	std::vector<ValuePtr> zero;
	zero.reserve(nout);
	for(size_t idx = 0; idx < nout; ++idx)
		zero.emplace_back(Value::create(0.0f));
	return zero;
}


class Neuron{
	private:
		Tensor weights;
		ValuePtr bias = Value::create(0.0, "");
		const ActivationType act;
	
	public:
		Neuron(size_t nin, const ActivationType act) : weights(randomWeights(nin), {nin}), act(act){}
	
	
	void zeroGrad(){
		for(const auto& weight : weights.data())
			weight->grad = 0;
		bias->grad = 0;
	}
	
	 std::vector<ValuePtr> Parameter() const{
		std::vector<ValuePtr> out;
		out.reserve(weights.numel() + 1);
		const auto& weightData = weights.data();
		out.insert(out.end(), weightData.begin(), weightData.end());
		out.push_back(bias);
		
		return out;
	}
	
	void printParameters() const{
		printf("No. of Parameters: %zu\n", weights.numel());
		for(const auto& params: weights.data())
			printf("%f, %f\n", params->data, params->grad);
		printf("%f, %f\n", bias->data, bias->grad);
		printf("\n");
	}
	
	ValuePtr operator()(const Tensor& x) const{
		if(weights.numel() != x.numel())
			throw std::invalid_argument("Inputs & Neuron Weights need to be of the same size");
			
		ValuePtr sum = Value::create(0.0);
		for(size_t idx = 0; idx < weights.numel(); ++idx)
			sum = Value::add(sum, Value::multiply(weights.at({idx}), x.at({idx})));
		
		sum = Value::add(sum, bias);
		
		const auto& activationFun = Activation::actfunc.at(act);
		
		return activationFun(sum);
	}
};

class Layer{
	private: 
		Tensor weights;
		Tensor bias;
		const ActivationType act;
		size_t nin;
		size_t nout;
		
	public:
		Layer(size_t nin, size_t nout, const ActivationType& act) : weights(randomWeights(nin * nout), {nin, nout}), bias(zeroBiases(nout), {1, nout}), nin(nin), nout(nout), act(act){}
		
		Tensor operator()(const Tensor& x){
			if(x.getShape()[1] != nin)
				throw std::invalid_argument("Input tensor is not equal to layer's input size");
			Tensor y = x.matmul_2d(weights) + bias;
			
			const auto& activationFun = Activation::actfunc.at(act);
			
			std::vector<ValuePtr> a;
			a.reserve(y.numel());
			
			for(const ValuePtr& v : y.data())
				a.push_back(activationFun(v));
			
			return Tensor(a, y.getShape());
		}
		
		void zeroGrad(){
			for(const ValuePtr& v : weights.data())
				v->grad = 0;
			for(const ValuePtr& v : bias.data())
				v->grad = 0;
		}
		
		std::vector<ValuePtr> Parameter() const{
			std::vector<ValuePtr> params;
			params.reserve(weights.numel() + bias.numel());
			
			params.insert(params.end(), weights.data().begin(), weights.data().end());
			params.insert(params.end(), bias.data().begin(), bias.data().end());
			
			return params;
		}
		
		void printParameters(){
			printf("\nNo. of Parameters : %zu\n", weights.numel() + bias.numel());
			for(size_t j = 0; j < nout; ++j){
				std::cout << "Neuron : " << j << "\nWeights : [" << std::endl;
				for(size_t i = 0; i < nin; ++i){
					std::cout <<  weights.at({i, j})->data <<  ", ";
				}
				std::cout << "Bias : [" << bias.at({0, j})->data << std::endl;
			}			
		}
		
};

/*
class Layer{
	public:
		std::vector<Neuron> neurons;
		size_t nin;
		size_t nout;
		Layer(size_t nin, size_t nout, const ActivationType& act) : nin(nin), nout(nout){
			neurons.reserve(nout);
			for(size_t idx = 0; idx < nout; ++idx)
				this->neurons.emplace_back(nin, act);
		} 
		
		Tensor operator()(const Tensor& x){
			if((x.ndim() == 1) && (x.numel() == nin)){
				std::vector<ValuePtr> out;
				out.reserve(neurons.size());
				for(const auto& neuron : neurons)
					out.emplace_back(neuron(x));

				return Tensor(out, {nout});
			}
			
			if((x.ndim() == 2) && (x.getShape()[1] == nin)){
				std::vector<ValuePtr> out;
				
				size_t batches = x.getShape()[0];
				
				out.reserve(batches * nout);
				
				for(size_t i = 0; i < batches; ++i){
					auto sample = x.slice({i});
					for(const auto& neuron : neurons)
						out.emplace_back(neuron(sample));
				}
				
				return Tensor(out, {batches, nout});
			}
			
			throw std::invalid_argument("Incompatible Input Given");
		}
		
		void zeroGrad(){
			for(auto& neuron : neurons)
				neuron.zeroGrad();
		}
		
		std::vector<ValuePtr> Parameter() const{
			std::vector<ValuePtr> params;
			params.reserve(nout * (nin + 1));
			for(const auto& neuron : neurons)
				for(const auto& p : neuron.Parameter())
					params.push_back(p);
					
			return params;
		}
		
		void printParameters(){
			const auto& params = this->Parameter();
			printf("No. of Parameters: %zu\n", params.size());
			for(const auto& p: params){
				std::cout << &p << " ";
				printf("%f, %f\n", p->data, p->grad);
			}
			printf("\n");	
		}
		
};
*/
/*
class Tensor{
private:
	std::vector<std::vector<ValuePtr>> tensor;

public:
	Tensor(std::initializer_list<float> input){
		for(const auto& elem : input){
			std::vector<ValuePtr> subtensor;
			subtensor.push_back(Value::create(elem));
			tensor.push_back(subtensor);
		}
	}
	
	auto begin(){
		return tensor.begin();
	}
	
	auto end(){
		return tensor.end();
	}
	
	auto begin() const{
		return tensor.begin();
	}
	
	auto end() const{
		return tensor.end();
	}
	
	//i th row
	std::vector<ValuePtr> operator[](const size_t idx) const{
		if(idx >= tensor.size())
			throw std::invalid_argument("Index out of bound");
		
		return tensor[idx];
	}
	
	//j th row
	ValuePtr at(const size_t idx, const size_t jdx) const{
		if(idx >= tensor.size() || jdx >= tensor[idx].size())
			throw std::invalid_argument("Index out of bound");
		
		return tensor[idx][jdx];
	}
	
	auto reset(){
		tensor.clear();
	}
	
	auto push_back(const std::vector<ValuePtr> elems){
		std::vector<ValuePtr> subtensor;
		std::copy(elems.begin(), elems.end(), std::back_inserter(subtensor));
		this->tensor.emplace_back(subtensor);
	}
	
	size_t size() const{
		return tensor.size();
	}
	
	void zeroGrad(){
		for(auto& subtensor : tensor)
			for(auto& value: subtensor)
				value->grad = 0;
	}
};
*/

class MLP{
private:
	std::vector<size_t> sizes;
	std::vector<Layer> layers;
	float learning_rate;
	size_t t = 0;
	
public:	
	MLP(size_t nin, std::vector<size_t> nouts, const float learning_rate) : learning_rate(learning_rate){
		sizes.reserve(nouts.size() + 1);
		sizes.push_back(nin);
		std::copy(nouts.begin(), nouts.end(), std::back_inserter(sizes));
		
		for(size_t idx = 0; idx < sizes.size() - 1; ++idx){
			layers.emplace_back(sizes[idx], sizes[idx+1], ActivationType::SIGMOID);
		}
	}
	
	std::vector<ValuePtr> Parameter() const{
		std::vector<ValuePtr> params;
		if (params.empty())
			for(const auto& layer : this->layers)
				for(const auto& p : layer.Parameter())
					params.push_back(p);
					
		return params;
	}
	
	void printParameters(){
			const auto& params = this->Parameter();
			printf("No. of Parameters: %zu\n", params.size());
			for(const auto& p: params){
				std::cout << &p << " ";
				printf("%f, %f\n", p->data, p->grad);
			}
			printf("\n");	
		}
		
	void zeroGrad(){
			for(auto& layer : this->layers)
				layer.zeroGrad();
		}
	
	void gradientDescent(){
		for(auto& p : this->Parameter())
			p->data += ((-learning_rate) * p->grad);
	}
	
	void gradientDescentwithMomentum(const float Beta){
		for(auto& p : this->Parameter()){
			p->velocity = (Beta * p->velocity) + (learning_rate * p->grad);
			p->data = p->data - p->velocity;
 		}
	}
	
	void beginNAG(const float Beta){
		for(auto& p : this->Parameter()){
			p->data -= (Beta * p->velocity);
		}
	}
	
	void endNAG(const float Beta){
		for(auto& p : this->Parameter()){
			p->data += (Beta * p->velocity);
		}
	}
	
	void NAG(const float Beta){
		for(auto& p : this->Parameter()){
			p->velocity = (Beta * p->velocity) + (learning_rate * p->grad);
			p->data = p->data - p->velocity;
		}
	}
	
	void Adagrad(){
		const float e = 1e-8f;
		for(auto& p : this->Parameter()){
			p->r = p->r + (p->grad * p->grad);
			p->data += ((-learning_rate/(std::sqrt(p->r) + e)) * p->grad);
		}
	}
	
	void RMSprop(){
		const float e = 1e-8f;
		const float beta = 0.95;
		for(auto& p : this->Parameter()){
			p->r = (beta * p->r) + ((1 - beta) * (p->grad * p->grad));
			p->data += ((-learning_rate/(std::sqrt(p->r) + e)) * p->grad);
		}
	}
	
	void Adam(const float Beta1 = 0.9, const float Beta2 = 0.999){
		const float e = 1e-8f;
		++t;
		for(auto& p : this->Parameter()){
			p->m = (Beta1 * p->m) + ((1 - Beta1) * (p->grad));
			p->r = (Beta2 * p->r) + ((1 - Beta2) * (p->grad * p->grad));
			float m_bc = p->m / (1 - std::pow(Beta1, t));
			float r_bc = p->r / (1 - std::pow(Beta2, t));
			p->data += ((-learning_rate * m_bc)/(std::sqrt(r_bc) + e));
		}
		
	}
	
	Tensor operator()(const Tensor& input){
		Tensor x = input;
		
		for(auto& layer : layers){
			Tensor y = layer(x);
			x = y;
		}
		
		return x;
	}
	
};

/*
int main(){
	Tensor input({0.266667, 0.266667, 0.294118, 0.301961, 0.333333, 0.152941, 0.215686, 0.290196, 0.282353, 0.298039, 0.313726, 0.309804, 0.317647, 0.34902, 0.32549, 0.266667, 0.266667, 0.294118, 0.301961, 0.333333, 0.152941, 0.215686, 0.290196, 0.282353, 0.298039, 0.313726, 0.309804, 0.317647, 0.34902, 0.32549, 0.333333, 0.333333, 0.529412, 0.666275, 0});

	Tensor output({0.266667, 0.266667, 0.294118, 0.301961, 0.333333, 0.152941, 0.215686, 0.290196, 0.282353, 0.298039, 0.313726, 0.309804, 0.317647, 0.34902, 0.32549, 0.266667, 0.266667, 0.294118, 0.301961, 0.333333, 0.152941, 0.215686, 0.290196, 0.282353, 0.298039, 0.313726, 0.309804, 0.317647, 0.34902, 0.32549, 0.333333, 0.333333, 0.529412, 0.666275, 0});
	
	
	const float LearningRate = 0.01;
	
	size_t nwSize = input[0].size();
	
	std::cout << "Input n/w size: " << nwSize << std::endl;
	
	auto mlp = std::unique_ptr<MLP>(new MLP(nwSize ,{16, 16, nwSize} , LearningRate));
	
	auto start_time_point = std::chrono::high_resolution_clock::now();
	
	Tensor ypred({});
	
	for(size_t idx = 0; idx < 1000000; ++idx){
		std::cout << "//////////////////////////////////////////////////////////////////////////" << std::endl;
		
		input.zeroGrad();
		
		auto loss = Value::create(0.0f);
		
		for(const auto& inp : input)
			ypred.push_back((*mlp)(inp));
			
		//Loss
		
		for(size_t idx = 0; idx < output[0].size(); ++idx){
			auto c = Value::substract(output.at(0, idx), ypred.at(0, idx));
			auto d = Value::multiply(c, c);
			
			loss = Value::add(loss, d);
		}
		
		mlp->zeroGrad();
		
		loss->backprop();
		
		mlp->Adam(0.9);
		
		std::cout << "Iteration: " << idx << " " << "Loss: " << loss->data << " , " << Value::currentID << std::endl;
		
		ypred.reset();
	}
}
*/



Tensor getSample(const Tensor& x, size_t sampleIndex) {
    if(sampleIndex >= x.getShape()[0])
        throw std::invalid_argument("Cannot access elements outside the sample");

    size_t features = x.getShape()[1];

    std::vector<ValuePtr> data;
    data.reserve(features);

    for(size_t idx = 0; idx < features; ++idx)
        data.push_back(x.at({sampleIndex, idx}));

    return Tensor(data, {features});
}



/*
int main(){
	std::vector<ValuePtr> data;

	for(float i = 1; i <= 24; ++i)
   	data.push_back(Value::create(i));

	Tensor x(data, {2, 3, 4});	
	
	Tensor y = x.reshape({6, 4});
	
	std::cout << "Original:\n";
	
	std::cout << "Shape: " << printVector(x.getShape())<< std::endl;
	
	std::cout << "Stride: " << printVector(x.getStride())<< std::endl;

	for(const auto& v : x.data())
   	std::cout << v->data << " ";

	std::cout << "\nReshaped:\n";
	
	std::cout << "Shape: " << printVector(y.getShape())<< std::endl;
	
	std::cout << "Stride: " << printVector(y.getStride())<< std::endl;

	for(const auto& v : y.data())
   	std::cout << v->data << " ";

	std::cout << "\n\nElement in reshaped tensor at position (5, 3) is: " << (y.at({5, 3})->data) << std::endl;
	
	Tensor x1 = x.slice({0});
	Tensor x2 = x.slice({1});
	Tensor x3 = x.slice({1, 2});
	
	std::cout << "Slice Test" << std::endl;
	
	for(const auto& v : x1.data())
   	std::cout << v->data << " ";
	std::cout << "Shape: " << printVector(x1.getShape())<< std::endl;
	std::cout << "Stride: " << printVector(x1.getStride())<< std::endl;
	for(const auto& v : x2.data())
   	std::cout << v->data << " ";
	std::cout << "Shape: " << printVector(x2.getShape())<< std::endl;
	std::cout << "Stride: " << printVector(x2.getStride())<< std::endl;
	for(const auto& v : x3.data())
   	std::cout << v->data << " ";
	std::cout << "Shape: " << printVector(x3.getShape())<< std::endl;
	std::cout << "Stride: " << printVector(x3.getStride())<< std::endl;
}
*/
/*
int main(){
	std::vector<ValuePtr> data;

	for(float i = 1; i <= 12; ++i)
   	data.push_back(Value::create(i));

	// 3 samples, each with 4 features
	Tensor batch(data, {3, 4});

	Layer layer(4, 2, ActivationType::SIGMOID);

	// Batched forward pass
	Tensor batchOutput = layer(batch);

	// Individual forward passes
	Tensor sample0 = batch.slice({0});
	Tensor sample1 = batch.slice({1});
	Tensor sample2 = batch.slice({2});

	Tensor output0 = layer(sample0);
	Tensor output1 = layer(sample1);
	Tensor output2 = layer(sample2);

	std::cout << "Batched output:\n";

	for(const auto& v : batchOutput.data())
   	std::cout << v->data << " ";

	std::cout << "\n\nIndividual outputs:\n";

	for(const auto& v : output0.data())
   	std::cout << v->data << " ";

	std::cout << "\n";

	for(const auto& v : output1.data())
   std::cout << v->data << " ";

	std::cout << "\n";

	for(const auto& v : output2.data())
   	std::cout << v->data << " ";

	std::cout << "\n";
}
*/

int main(){
	std::vector<ValuePtr> indata;
	
	indata.push_back(Value::create(0.0f));
	indata.push_back(Value::create(0.0f));
	
	indata.push_back(Value::create(0.0f));
	indata.push_back(Value::create(1.0f));
	
	indata.push_back(Value::create(1.0f));
	indata.push_back(Value::create(0.0f));
	
	indata.push_back(Value::create(1.0f));
	indata.push_back(Value::create(1.0f));
	
	Tensor input(indata, {4,2});
	
	std::vector<ValuePtr> outdata;
	
	outdata.push_back(Value::create(0.0f));
	
	outdata.push_back(Value::create(1.0f));
	
	outdata.push_back(Value::create(1.0f));
	
	outdata.push_back(Value::create(0.0f));
	
	Tensor output(outdata, {4,1});
	
	std::vector<size_t> a = {2, 3};
	std::vector<size_t> b = {4};
	
	//std::cout << printVector(Tensor::broadcastShape({4}, {})) << std::endl;
	
	std::cout << printVectorData((input.broadcastAdd(output)).data()) << std::endl;
	std::cout << printVectorData((input - output).data()) << std::endl;
	
	std::vector<ValuePtr> data1;

	for(float i = 1; i <= 24; ++i)
   	data1.push_back(Value::create(i));

	Tensor x(data1, {2, 3, 4});	
	
	std::vector<ValuePtr> data2;

	for(float i = 1; i <= 12; ++i)
   	data2.push_back(Value::create(i));

	Tensor y(data2, {3, 4});
	
	std::vector<ValuePtr> data3;
	
	for(float i = 1; i <= 4; ++i)
   	data3.push_back(Value::create(i));

	Tensor z(data3, {1, 4});
	
	
	
	for(size_t i = 0; i < x.getShape()[0]; i++)
		for(size_t j = 0; j < x.getShape()[1]; j++)
			for(size_t k = 0; k < x.getShape()[2]; k++){
				//x.broadcastIndexMap(std::vector<size_t> idx{i, j, k});
				//std::cout << printVector(std::vector<size_t> {i, j, k}) << std::endl;
				std::cout << printVector(x.broadcastIndexMap(std::vector<size_t> {i, j, k})) << std::endl;
			}
	
	std::cout << std::endl;
	std::cout << std::endl;
	
	for(size_t i = 0; i < x.getShape()[0]; i++)
		for(size_t j = 0; j < x.getShape()[1]; j++)
			for(size_t k = 0; k < x.getShape()[2]; k++)
				//y.broadcastIndexMap(std::vector<size_t> idx{i, j, k});
				std::cout << printVector(z.broadcastIndexMap(std::vector<size_t> {i, j, k})) << std::endl;
			
			
	
	std::cout << printVectorData((x.broadcastAdd(z)).data()) << std::endl;
	std::cout << printVectorData((x.broadcastAddExp(z)).data()) << std::endl;
	std::cout << printVectorData((x + z).data()) << std::endl;
	
	
	std::vector<ValuePtr> data4;

	for(float i = 1; i <= 6; ++i)
   	data4.push_back(Value::create(i));
	
	Tensor m(data4, {2, 3});
	
	std::vector<ValuePtr> data5;

	for(float i = 1; i <= 6; ++i)
   	data5.push_back(Value::create(i));

	Tensor n(data5, {3, 2});
	
	std::cout << printVectorData((m.matmul_2d(n)).data()) << std::endl;
	std::cout << printVector((m.transpose_2d()).getShape()) << std::endl;
	std::cout << printVector((m.transpose_2d()).getStride()) << std::endl;
	std::cout << printVector((n.transpose_2d()).getShape()) << std::endl;
	std::cout << printVector((n.transpose_2d()).getStride()) << std::endl;

	std::cout << "\n";
	
	Tensor trans1 = m.transpose_2d();
	for(size_t i = 0; i < trans1.numel(); ++i){
   	std::cout << trans1.at(trans1.flat_to_multidimension(i))->data << " ";
	}
	
	std::cout << "\n";
	
	Tensor trans2 = n.transpose_2d();
	for(size_t i = 0; i < trans2.numel(); ++i){
   	std::cout << trans2.at(trans2.flat_to_multidimension(i))->data << " ";
	}
	
	std::cout << "\n";
	std::cout << m.iscontiguous() << std::endl;
	
	std::cout << trans1.iscontiguous() << std::endl;
	
	Tensor trans = trans1.contiguous();
	
	std::cout << trans.iscontiguous() << std::endl;
	
	std::cout << printVectorData(trans.data()) << std::endl;
	
	
	std::vector<ValuePtr> data6;

	for(float i = 1; i <= 4; ++i)
   	data6.push_back(Value::create(i));
	
	Tensor sum(data6, {2,2});
	
	std::cout << printVectorData(sum.sum().data()) << std::endl;
	std::cout << printVectorData(m.sum(1).data()) << std::endl;
	
	std::vector<ValuePtr> data7;

	for(float i = 1; i <= 12; ++i)
   	data7.push_back(Value::create(i));
	
	Tensor matmul1(data7, {2, 1, 2, 3});
	
	std::vector<ValuePtr> data8;

	for(float i = 1; i <= 18; ++i)
   	data8.push_back(Value::create(i));
	
	Tensor matmul2(data8, {3, 3, 2});
	
	std::cout << printVectorData((matmul1.matmul(matmul2)).data()) << std::endl;
	std::cout << printVector((matmul1.matmul(matmul2)).getShape()) << std::endl;
	
	Tensor result = matmul1.matmul(matmul2);
	Tensor loss = result.sum();
	loss.backprop();

	// Print gradients to ensure backpropagation flowed to both inputs
	std::cout << "matmul1 gradients:\n";
	for (const auto& v : matmul1.data()) {
   	std::cout << v->grad << " ";
	}
	std::cout << "\n\nmatmul2 gradients:\n";
	for (const auto& v : matmul2.data()) {
   	std::cout << v->grad << " ";
	}
	std::cout << std::endl;
	/*
	std::vector<ValuePtr> data9;

	for(float i = 1; i <= 3; ++i)
   	data9.push_back(Value::create(i));
	
	Tensor u(data9, {3});
	Tensor v(data9, {3});
	Tensor dot = u.matmul(v); // shape: [1]
	std::cout << printVectorData(dot.data()) << "\t" << printVector(dot.getShape()) <<std::endl;
	
	Tensor q(data9, {3});
	Tensor M(data5, {3, 2});
	Tensor qm = q.matmul(M); // shape: [2]
	std::cout << printVectorData(qm.data()) << "\t" << printVector(qm.getShape()) <<std::endl;
	
	std::vector<ValuePtr> data10;

	for(float i = 1; i <= 2; ++i)
   	data10.push_back(Value::create(i));
	
	Tensor BM(data7, {2, 3, 2});
	Tensor w(data10, {2});
	Tensor bmv = BM.matmul(w); // shape: [2, 3]
	std::cout << printVectorData(bmv.data()) << "\t" << printVector(bmv.getShape()) <<std::endl;
	*/
	
	// 1. Vector dot product: [3] @ [3] -> [1]
	Tensor v1({Value::create(1), Value::create(2), Value::create(3)}, {3});
	Tensor v2({Value::create(4), Value::create(5), Value::create(6)}, {3});
	Tensor dot = v1.matmul(v2);
	assert(dot.getShape() == std::vector<size_t>({1}));
	assert(dot.data()[0]->data == 32.0f); // 1*4 + 2*5 + 3*6 = 32

	// 2. Vector @ Matrix: [3] @ [3, 2] -> [2]
	std::vector<ValuePtr> m_vals;
	for(float i = 1; i <= 6; ++i) m_vals.push_back(Value::create(i));
		Tensor M(m_vals, {3, 2});
		Tensor vm = v1.matmul(M);
		assert(vm.getShape() == std::vector<size_t>({2}));

	// 3. Matrix @ Vector: [2, 3] @ [3] -> [2]
	Tensor mv = M.transpose_2d().matmul(v1); // [2, 3] @ [3] -> [2]
	assert(mv.getShape() == std::vector<size_t>({2}));

	
	/*
	auto mlp = MLP(2, {4, 1}, 0.1f);
	
	for(size_t i = 0; i < 10000; ++i){
		mlp.zeroGrad();
		
		Tensor pred = mlp(input);
		
		ValuePtr loss = Value::create(0.0f);
		
		for(size_t i = 0; i < pred.getShape()[0]; i++){
			ValuePtr c = Value::substract(pred.at({i, 0}), output.at({i, 0}));
			ValuePtr d = Value::multiply(c, c);
		
			loss = Value::add(loss, d);
		}
		
		loss->backprop();
		
		mlp.Adam();
		
		if(i % 100 == 0)
			std::cout << "Iteration: " << i << " " << "Loss: " << loss->data << " , " << Value::currentID << std::endl;
	}
	
	std::cout << "\n\n\t Final Predictions\n\n" << std::endl;
	
	Tensor pred = mlp(input);
	for(auto v : pred.data())
		std::cout << v->data << std::endl;
	*/
}	
