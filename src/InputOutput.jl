import Base: show, summary
show(io::IO, x::ScalarOperator{F}) where {F} = print(io, "op $(x.name) ($F)}");
show(io::IO, x::BroadcastedOperator{F}) where {F} = begin 
    print(io, "op $(x.name), ($F)");
    #print(io, "\n ┣━ ^ $(x.output)");
    #print(io, "\n ┗━ ∇ $(x.gradient)");
end
show(io::IO, x::Constant) = print(io, "const $(x.output)");
show(io::IO, x::Variable) = begin
    print(io, "var $(x.name)");
    print(io, "\n ┣━ ^ $(x.output)");
    print(io, "\n ┗━ ∇ $(x.gradient)");
end
