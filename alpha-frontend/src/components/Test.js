import React from 'react';

class PostRequest extends React.Component {
    constructor(props) {
        super(props);

        this.state = {
            value: '',
            postId: null
        };
    }

    handleTickerChange(event) {
      this.setState({value: event.target.value});
    };

    componentDidMount() {
        // Simple POST request with a JSON body using fetch
        const requestOptions={
            method:"POST",
            headers:{"Content-Type":"Application/json"},
            body:JSON.stringify({
                ticker:this.state.value
            }),
        };
        fetch('http://127.0.0.1:8000/analysis/create-result', requestOptions)
            .then(response => response.json())
            .then(data => this.setState({ postId: data.result }));
    }

    render() {
        const { postId } = this.state;
        return (
            <div className="card text-center m-3">
                <h5 className="card-header">Simple POST Request</h5>
                <div className="card-body">
                    Returned Id: {postId}
                </div>
            </div>
        );
    }
}

export default PostRequest;